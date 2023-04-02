import torch
import wandb
import pytorch_lightning.loggers
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.cli import LightningCLI

# needed for LightningCLI
import models


def get_ground_truth_and_preds(
        trainer,
        model,
        dataloader,
):
    ret = trainer.predict(model, dataloaders=dataloader)
    features = torch.cat([f[0][1] for f in ret])
    Y = torch.cat([f[1] for f in ret])
    S_shared, S_indiv = model.make_S_shared_S_indiv(Y)
    Y_shared = model.make_Y_shared(Y)

    return features, Y, S_shared, S_indiv, Y_shared


def eval_model(model, trainer):
    with torch.no_grad():
        features_val, Y_val, _, _, _ = get_ground_truth_and_preds(
            trainer=trainer,
            model=model,
            dataloader=model.val_dataloader(),
        )
        features_test, _, S_shared_test, S_indiv_test, Y_shared_test = get_ground_truth_and_preds(
            trainer=trainer,
            model=model,
            dataloader=model.test_dataloader(),
        )
        metrics_dcid = model.compute_dcid_metrics(
            features_val=features_val,
            features_test=features_test,
            Y_val=Y_val,
            S_shared_test=S_shared_test,
            S_indiv_test=S_indiv_test,
            Y_shared_test=Y_shared_test,
        )

        return metrics_dcid


def cli_main():
    logger = pytorch_lightning.loggers.WandbLogger(project='dcid-toy', save_dir='logs')

    cli = LightningCLI(
        save_config_overwrite=True,
        run=False,
        trainer_defaults={
            'logger': logger,
            'gpus': int(torch.cuda.is_available()),
            'enable_checkpointing': True,
        }
    )

    seed = cli.config['seed_everything']
    if 'seed_everything' not in logger.experiment.config or logger.experiment.config['seed_everything'] is None:
        logger.experiment.config['seed_everything'] = seed
    seed_everything(seed)

    cli.trainer.fit(cli.model)

    # evaluate with DCID metrics
    metrics_dcid = eval_model(model=cli.model, trainer=cli.trainer)
    wandb.log({'metrics_dcid': wandb.Table(dataframe=metrics_dcid)})


if __name__ == '__main__':
    cli_main()
