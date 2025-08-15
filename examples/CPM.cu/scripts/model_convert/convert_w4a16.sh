Model_Path=/yourpath/minicpm4_mupformat
Quant_Path=/yourpath/minicpm4_autogptq
Output_Path=/yourpath/minicpm4_marlin


python model_convert/convert_w4a16.py \
    --model-path $Model_Path \
    --quant-path $Quant_Path \
    --output-path $Output_Path 