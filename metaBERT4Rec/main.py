import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def train():
    #export_root = setup_train(args)
    export_root = args.export
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    
    if args.load:
        best_model = torch.load(args.load).get('model_state_dict') 
        if args.full:
            del best_model['out.weight']
            del best_model['out.bias']
            model_dict = model.state_dict()
            model_dict.update(best_model) 
            model.load_state_dict(model_dict)
        else:
            model_dict = model.state_dict()
            model_dict['bert.ue.token.weight'] = best_model['bert.ue.token.weight']
            model_dict['bert.ue.position.pe.weight'] = best_model['bert.ue.position.pe.weight']
            if args.kg:
                model_dict['bert.dir.token.weight'] = best_model['bert.dir.token.weight']
                model_dict['bert.dir.position.pe.weight'] = best_model['bert.dir.position.pe.weight']
                model_dict['bert.act.token.weight'] = best_model['bert.act.token.weight']
                model_dict['bert.act.position.pe.weight'] = best_model['bert.act.position.pe.weight']
                
            model.load_state_dict(model_dict)
        
    #model.load_state_dict(best_model)
    
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    trainer.test()


if __name__ == '__main__':
    #torch.manual_seed(34567)#34567 
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
