import wandb

def upload_files_to_wandb(project_name, run_name, model_weights_path=None, config=None):
    # WandB 초기화
    wandb.init(
        project=project_name,
        name=run_name,
        config=config
    )

    # data_provider 파일 업로드
    data_provider_artifact = wandb.Artifact('data_provider_scripts', type='code')
    data_provider_files = [
        './data_provider/data_factory.py',
        './data_provider/data_loader.py'
    ]
    for file_path in data_provider_files:
        data_provider_artifact.add_file(file_path)
    wandb.log_artifact(data_provider_artifact)

    # layers 파일 업로드
    layers_artifact = wandb.Artifact('layers_scripts', type='code')
    layers_files = [
        './layers/Autoformer_EncDec.py',
        './layers/Embed.py',
        './layers/PatchTST_backbone.py',
        './layers/PatchTST_layers.py',
        './layers/RevIN.py',
        './layers/SelfAttention_Family.py',
        './layers/Transformer_EncDec.py'
    ]
    for file_path in layers_files:
        layers_artifact.add_file(file_path)
    wandb.log_artifact(layers_artifact)

    # models 파일 업로드 (PatchTST 모델 파일)
    models_artifact = wandb.Artifact('models_scripts', type='code')
    models_files = [
        './models/PatchTST.py'
    ]
    for file_path in models_files:
        models_artifact.add_file(file_path)
    wandb.log_artifact(models_artifact)

    # run_longExp.py 파일 업로드
    run_script_artifact = wandb.Artifact('run_script', type='code', description="Main training script for long experiments")
    run_script_artifact.add_file('./run_longExp.py')
    wandb.log_artifact(run_script_artifact)

    # 가장 최근 모델 가중치 파일 업로드 (model_latest.pth)
    if model_weights_path:
        model_artifact = wandb.Artifact(
            name='latest_model_weights',
            type='model',
            description='Most recent model weights during training'
        )
        model_artifact.add_file(model_weights_path)
        wandb.log_artifact(model_artifact)

    print("WandB upload complete.")
