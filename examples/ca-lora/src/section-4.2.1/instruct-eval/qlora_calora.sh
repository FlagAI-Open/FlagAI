python main.py mmlu --model_name qlora_calora --model_path nop > qlora_calorammlu.log
python main.py humaneval --model_name qlora_calora --model_path nop > qlora_calorahumaneval.log
python main.py bbh --model_name qlora_calora --model_path nop  > qlora_calorabbh.log
python main.py drop --model_name qlora_calora --model_path nop > qlora_caloradrop.log