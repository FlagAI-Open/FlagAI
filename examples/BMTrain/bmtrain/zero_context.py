import torch
from . import nccl
from .global_var import config
from .synchronize import wait_loader


class ZeroContext:
    """ZeroContext is a helper class to Gather parameters before module forward and reduce scatter
    gradients after module backward.

    Args:
        block (BLock): Input Block.
        ctx_dict (dict): block._layer_dict.
        pipe (bool): True if use pipe parallel.

    """

    def __init__(self, block: "Block", ctx_dict: dict = None, pipe=False) -> None:
        self.block = block
        self.ctx_dict = ctx_dict
        self._param_buffer = {}
        self._grad_buffer = {}
        self._param_tensor = {}
        self._grad_tensor = {}
        self._need_release = False

    def enter(self, flag=0, requires_grad=False):
        """
        Gather parameters before module forward and init grad buffer before backward.
        """
        if self.block._ready:
            return
        self.block._ready = True
        self._need_release = True

        wait_loader()
        with torch.cuda.stream(config["load_stream"]):
            for kw, val in self.block._storage_info.items():
                assert self.block._storage_params[kw].is_cuda
                assert kw not in self._grad_buffer
                assert kw not in self._param_buffer
                local_param = self.block._storage_params[kw]

                storage_type = local_param.storage_type()
                if flag != 2:
                    self._param_buffer[kw] = storage_type(
                        val["partition_size"] * val["world_size"]
                    )
                    self._param_tensor[kw] = torch.tensor(
                        [],
                        dtype=self._param_buffer[kw].dtype,
                        device=self._param_buffer[kw].device,
                    ).set_(self._param_buffer[kw])

                if requires_grad and local_param.requires_grad:
                    self._grad_buffer[kw] = storage_type(
                        val["partition_size"] * val["world_size"]
                    )
                    self._grad_tensor[kw] = (
                        torch.tensor(
                            [],
                            dtype=self._grad_buffer[kw].dtype,
                            device=self._grad_buffer[kw].device,
                        )
                        .set_(self._grad_buffer[kw])
                        .zero_()
                    )
            if flag != 2:
                nccl.groupStart()
                for kw, val in self.block._storage_info.items():
                    nccl.allGather(
                        self.block._storage_params[kw].storage(),
                        self._param_buffer[kw],
                        val["zero_comm"],
                    )
                nccl.groupEnd()

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(config["load_stream"])

        # set wait stream for each storage
        for kw in self.block._storage_info.keys():
            if flag != 2:
                self._param_tensor[kw].record_stream(current_stream)
            if requires_grad and kw in self._grad_tensor:
                self._grad_tensor[kw].record_stream(current_stream)

        # update parameters in block
        for param in self.block._param_info:
            kw_name = param["kw_name"]
            offset = param["offset"]
            shape = param["shape"]

            if flag != 2:
                dtype = self._param_buffer[kw_name].dtype
                device = self._param_buffer[kw_name].device
                param["parameter"].data = torch.tensor(
                    [], dtype=dtype, device=device
                ).set_(self._param_buffer[kw_name], offset, shape)
            else:
                dtype = param["parameter"].data.dtype
                device = param["parameter"].data.device
                param["parameter"].data = torch.tensor(
                    [], dtype=dtype, device=device
                ).set_(self.ctx_dict[kw_name], offset, shape)

            if (
                requires_grad
                and kw_name in self._grad_buffer
                and param["parameter"].requires_grad
            ):
                param["parameter"].grad = torch.tensor(
                    [], dtype=dtype, device=device
                ).set_(self._grad_buffer[kw_name], offset, shape)

    def __enter__(self):
        self.enter()

    def exit(self, flag=0, backward=False):
        """
        Reduce scatter gradients when backward and release all parameters from buffer to block_storge when forward is done.
        """
        if not self._need_release:
            return
        self._need_release = False
        self.block._ready = False
        if backward:
            for kw, val in self.block._storage_info.items():
                local_param = self.block._storage_params[kw]

                # accumulate previous gradient
                if local_param.requires_grad:
                    if local_param.grad is None:
                        grad_storage = val["storage_type"](
                            val["partition_size"]
                        )  # initialize gradient if not exist
                        local_param.grad = (
                            torch.tensor(
                                [], dtype=grad_storage.dtype, device=grad_storage.device
                            )
                            .set_(grad_storage)
                            .zero_()
                        )
                    else:
                        self._grad_tensor[kw][
                            val["begin"] : val["end"]
                        ] += local_param.grad

            current_stream = torch.cuda.current_stream()
            config["load_stream"].wait_stream(current_stream)  # wait for backward

            with torch.cuda.stream(config["load_stream"]):
                nccl.groupStart()
                for kw, val in self.block._storage_info.items():
                    local_param = self.block._storage_params[kw]

                    # scatter gradient
                    if local_param.requires_grad:
                        nccl.reduceScatter(
                            self._grad_buffer[kw],
                            local_param.grad.storage(),
                            "sum",
                            val["zero_comm"],
                        )
                nccl.groupEnd()

            # set wait stream for each storage
            for kw in self._grad_tensor.keys():
                # grads can not be freed until reduce ops finish
                self._grad_tensor[kw].record_stream(config["load_stream"])

        # Release all parameters from buffer to block_storge
        for param in self.block._param_info:
            kw_name = param["kw_name"]
            dtype = self.block._storage_params[kw_name].dtype
            device = self.block._storage_params[kw_name].device
            if "begin" not in param:
                param["parameter"].data = torch.tensor([], dtype=dtype, device=device)
                param["parameter"].grad = None
                continue
            begin = param["begin"]
            end = param["end"]
            param["parameter"].data = torch.tensor([], dtype=dtype, device=device).set_(
                self.block._storage_params[kw_name].storage(), begin, end
            )
            if (
                param["parameter"].requires_grad
                and self.block._storage_params[kw_name].grad is not None
            ):
                param["parameter"].grad = torch.tensor(
                    [], dtype=dtype, device=device
                ).set_(self.block._storage_params[kw_name].grad.storage(), begin, end)
        if flag == 1:
            for i in self._param_buffer:
                self.ctx_dict[i] = self._param_buffer[i]
        self._grad_tensor = {}
        self._param_tensor = {}
        self._grad_buffer = {}
        self._param_buffer = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        # reduce scatter gradients
        self.exit()
