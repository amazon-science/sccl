data_path=pre_data

mkdir -p processed_data

for file in $data_path/*.csv
do
    name=`echo $file | cut -d\. -f 1 | cut -d \/ -f 2`
    echo $name
    python AugData/nlpaug_explore.py --dataset $name --gpu_id 1 &
    sleep 10
done

# python AugData/nlpaug_explore.py --dataset agnews --gpu_id 0
# python AugData/nlpaug_explore.py --dataset biomedical --gpu_id 1
# python AugData/nlpaug_explore.py --dataset googlenews_S --gpu_id 1
# python AugData/nlpaug_explore.py --dataset googlenews_TS --gpu_id 1
# python AugData/nlpaug_explore.py --dataset googlenews_T --gpu_id 1
# python AugData/nlpaug_explore.py --dataset search_snippets --gpu_id 1
# python AugData/nlpaug_explore.py --dataset biomedical --gpu_id 1
# python AugData/nlpaug_explore.py --dataset tweet --gpu_id 1
# 
