from pathlib import Path

import numpy as np
import torch
import pytorch_lightning.loggers
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.cli import LightningCLI

# needed for LightningCLI
import models


def save_predictions(
        trainer,
        model,
        dataloader,
        df,
        preds_dir,
):
    preds_dir.mkdir(exist_ok=True)

    ret = trainer.predict(model=model, dataloaders=dataloader)
    features = torch.cat([f[0][1] for f in ret])
    Y = torch.cat([f[1] for f in ret])
    Y1, Y2 = Y[:, 0], Y[:, 1]

    np.save(preds_dir / 'features.npy', features)
    np.save(preds_dir / 'Y1.npy', Y1)
    np.save(preds_dir / 'Y2.npy', Y2)

    df.to_pickle(preds_dir / 'df.pkl')

    print(preds_dir.absolute())


def cli_main(wandb_project_name='dcid-brain-mri'):
    logger = pytorch_lightning.loggers.WandbLogger(project=wandb_project_name, save_dir='logs')

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

    # save predictions
    experiment_dir = Path(logger.experiment.dir)
    for dataloader, df, preds_name in (
            # PL's engine crashes when shuffle is True with trainer.predict()
            (cli.model.train_dataloader(shuffle=False), cli.model.df_train, 'train'),
            (cli.model.val_dataloader(), cli.model.df_val, 'val'),
            (cli.model.test_dataloader(), cli.model.df_test, 'test'),
    ):
        save_predictions(
            trainer=cli.trainer,
            model=cli.model,
            dataloader=dataloader,
            df=df,
            preds_dir=experiment_dir / preds_name,
        )


if __name__ == '__main__':
    cli_main()
