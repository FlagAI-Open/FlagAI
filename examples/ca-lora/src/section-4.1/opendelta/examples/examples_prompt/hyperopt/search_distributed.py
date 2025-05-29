import optuna
import argparse
import os
import shutil
import subprocess




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_type")
    parser.add_argument("--dataset")
    parser.add_argument("--model_name")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--cuda_ids", nargs='+', help="list")
    parser.add_argument("--mode", type=str, default="run", help="select from 'run' and 'read' ")
    parser.add_argument("--continue_study", type=int, default=0)
    parser.add_argument("--substudy_prefix", type=str, default="")
    parser.add_argument("--main_file_name", type=str)
    parser.add_argument("--num_trials", type=int)
    parser.add_argument("--pathbase", type=str, default="")
    parser.add_argument("--pythonpath", type=str, default="python")
    parser.add_argument("--plm_path_base", type=str, default="", help="The path where we cache the plms. Must be empty string or dir that ends with /")
    parser.add_argument("--datasets_load_from_disk", action="store_true")
    parser.add_argument("--datasets_saved_path", type=str)
    parser.add_argument("--repeat_time", type=int, default=1)
    args = parser.parse_args()


    pardir = ".".join([args.delta_type, args.dataset, args.model_name])
    if args.study_name is None:
        args.study_name = pardir
    else:
        args.study_name += pardir

    setattr(args, "output_dir", f"{args.pathbase}/outputs_search/{args.study_name}")



    if args.mode == "run":
        if args.continue_study==1:
            print("Continue study!")
        else:
            print("Creat new study!")

        if not os.path.exists(f"{args.output_dir}"):
            os.mkdir(f"{args.output_dir}")
        else:
            if not args.continue_study:
                user_cmd = "yes" #input("Detected existing study, are you sure to create new by removing old? [Yes/No]")

                while user_cmd.lower() not in ["yes", "no"]:
                    print("Please input Yes/No")
                    user_cmd = input("Detected existing study, are you sure to create new by removing old? [Yes/No]")

                if user_cmd.lower() == "no":
                    exit()
                shutil.rmtree(f"{args.output_dir}")
                os.mkdir(f"{args.output_dir}")

        try:
            study = optuna.create_study(study_name=args.study_name, storage=f"sqlite:///{args.study_name}.db")
        except optuna.exceptions.DuplicatedStudyError:
            if not args.continue_study:
                optuna.delete_study(study_name=args.study_name, storage=f"sqlite:///{args.study_name}.db")
                study = optuna.create_study(study_name=args.study_name, storage=f"sqlite:///{args.study_name}.db")
            else:
                pass # no need to create study

        tot_chunk_num = len(args.cuda_ids)

        subprocesses = []
        for id, cudas in enumerate(args.cuda_ids):
            if id+1 < tot_chunk_num:
                sub_n_trials = args.num_trials//tot_chunk_num
            else:
                sub_n_trials = args.num_trials//tot_chunk_num + args.num_trials%tot_chunk_num

            command =  f"{args.pythonpath} search_single.py "
            command += f"--cuda_id {cudas} "
            command += f"--model_name {args.model_name} "
            command += f"--dataset {args.dataset} "
            command += f"--delta_type {args.delta_type} "
            command += f"--study_name {args.study_name} "
            command += f"--optuna_seed 10{id} "
            command += f"--main_file_name {args.main_file_name} "
            command += f"--num_trials {sub_n_trials} "
            command += f"--pythonpath {args.pythonpath} "
            command += f"--pathbase {args.pathbase} "
            command += f"--repeat_time {args.repeat_time} "
            command += f"--plm_path_base {args.plm_path_base} "
            command += f"--datasets_saved_path {args.datasets_saved_path} "
            if args.datasets_load_from_disk:
                command += f"--datasets_load_from_disk "
            command += f"> {args.output_dir}/{args.substudy_prefix}{id}.log 2>&1"
            p = subprocess.Popen(command, cwd=f"{args.pathbase}", shell=True)
            subprocesses.append(p)
            print("id {} on cuda:{}, pid {}".format(id, cudas, p.pid))
            print(command)
            print()

        print("Wait for subprocesses to complete")
        exit_codes = [p.wait() for p in subprocesses]
        print("All complete!")

    elif args.mode == 'read':
        study = optuna.load_study(study_name=args.study_name, storage=f"sqlite:///{args.study_name}.db")
        trial = study.best_trial
        finished = (len(study.trials) == args.num_trials)
        print("total num_trials: {}, {}".format(len(study.trials), "Finished!" if finished else "Not finished..." ))
        print("average acc {}".format(-trial.value))
        print("best config {}".format(trial.params))

        best_trial_dir = trial.user_attrs["trial_dir"]
        shutil.copyfile(f"{best_trial_dir}/this_configs.json", f"{args.output_dir}/best_config.json")

        plot_history = optuna.visualization.plot_optimization_history(study)
        plot_slice = optuna.visualization.plot_slice(study)
        plot_contour = optuna.visualization.plot_contour(study, params=['learning_rate', 'batch_size_base'])
        plot_contour2 = optuna.visualization.plot_contour(study, params=['learning_rate', 'warmup_steps'])


        plot_history.write_image(f"{args.output_dir}/history.png")
        plot_slice.write_image(f"{args.output_dir}/slice.png")
        plot_contour.write_image(f"{args.output_dir}/contour.png")
        plot_contour2.write_image(f"{args.output_dir}/contour2.png")









