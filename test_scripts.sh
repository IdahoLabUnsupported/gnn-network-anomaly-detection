cd ~/cyberproject/internship2024/brueja1/project/
# python code/train.py --data_folderpath /home/zeeklogs \
#     --graph_folderpath ~/cyberproject/internship2024/brueja1/project/pickles \
#     --model_folderpath ~/cyberproject/internship2024/brueja1/project/models \
#     --logger TRACE
# python code/infer.py --data_folderpath /home/zeeklogs \
#     --graph_folderpath ~/cyberproject/internship2024/brueja1/project/pickles \
#     --model_folderpath ~/cyberproject/internship2024/brueja1/project/models \
#     --result_folderpath ~/cyberproject/internship2024/brueja1/project/results \
#     --logger TRACE
python code/evaluate.py --data_folderpath /home/zeeklogs \
    --graph_folderpath ~/cyberproject/internship2024/brueja1/project/pickles \
    --model_folderpath ~/cyberproject/internship2024/brueja1/project/models \
    --result_folderpath ~/cyberproject/internship2024/brueja1/project/results \
    --logger TRACE
