nproc_per_node=8
max_epoch=30

enable_aux_regloss=True
aux_regloss_weight=1.0

node_rank=0
master_addr="192.168.1.1"

cfg_file="run_scripts/train/regionblip_t5-xxl_freeze-qformer_regloss_pcqformer_A100.yaml"
run_output_dir="training_dir/regionblip_t5-xxl_freeze-qformer_regloss_pcqformer_A100__epoch-${max_epoch}__gpu${nproc_per_node}x4nodes__regloss-${enable_aux_regloss}-${aux_regloss_weight}"


if [ ${node_rank} == 0 ]; then
    if [ -e ${run_output_dir} ]; then
        echo "${run_output_dir} exist !"
        exit
    fi

    mkdir -p ${run_output_dir}
    cp $0 ${run_output_dir}/
    cp ${cfg_file} ${run_output_dir}/
fi


#TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
    --nnodes=4 --node_rank=${node_rank} --master_addr=${master_addr} --master_port=15556 \
    train.py \
    --cfg-path ${cfg_file} \
    --options run.output_dir=${run_output_dir} run.num_workers=${nproc_per_node} \
    run.max_epoch=${max_epoch} \
    model.enable_aux_regloss=${enable_aux_regloss} model.aux_regloss_weight=${aux_regloss_weight} \
    ${@:1} | tee -a ${run_output_dir}/log_all_${node_rank}.txt

echo ${run_output_dir}
