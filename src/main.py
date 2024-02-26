import click
from train import train



@click.command()
@click.option('--train', 'to_train', is_flag=True)
@click.option('--device', default='cuda', help='cuda or cpu')
@click.option('--dataset', help='dataset directory')
@click.option('--epochs', default=20)
@click.option('--batch-size', default=16)
@click.option('--lr', default=0.001)
@click.option('--get-attr-embs', 'to_get_attr_embs', is_flag=True)
def main(**kwargs):
    if kwargs['to_train']:
        train(dataset_path=kwargs['dataset'], device=kwargs['device'], epochs=kwargs['epochs'], batch_size=kwargs['batch_size'], lr=kwargs['lr'])
        
    if kwargs['to_get_attr_embs']:
        get_attr_embs(dataset_path=kwargs['dataset'], device=kwargs['device'], batch_size=kwargs['batch_size'])