data_path=pre_data

mkdir -p processed_data

for file in $data_path/*.csv
do
    name=`echo $file | cut -d\. -f 1 | cut -d \/ -f 2`
    echo $name
    python AugData/nlpaug_explore.py --dataset $name --gpu_id 1 &
    sleep 10
done