import os

model_names = ["se_resnext50", "resnext50"]
pool_types = ['avg', 'max']
scheduleres = ["cos", "step", "multi"]


for pool_type in pool_types:
    for scheduler in scheduleres:
        for model_name in model_names:
            os.system("python run.py \
                    --model_name {} \
                    --pool_type {} \
                    --scheduler {} \
                    ".format(model_name, pool_type, scheduler))