import os
import json

import torch
import numpy as np
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.datasets as datasets
import torch.nn.functional as F
import zipfile

def eval_single_dataset(image_classifier, dataset, args, classification_head):

    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    classification_head.eval()

    dataloader = get_dataloader(dataset,
                                is_train=False,
                                args=args,
                                image_encoder=image_enc)

    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        if args.current_epoch==-1:
            save_path='./result.txt'
            count=0
            save_file = open(save_path, 'w')
        for i, data in batched_data:

            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']

            logits = utils.get_logits(x, model, classification_head)
            # print('logits.shape',logits.shape)
            # print("logits[0].shape",logits[0].shape)
            # print("logits[0].topk(5)[1]",logits[0].topk(5)[1])
            # dataset.convert_id_to_name(logits[0].topk(5)[1].tolist())
            # assert False
            if args.current_epoch==-1:# 比赛测评写入文件
                
                for i in range(len(y.tolist())):
                    temp="image_"+str(int(y[i]))
                    name=temp+".jpg"
                    top_5=dataset.convert_id_to_name(logits[i].topk(5)[1].tolist())#一个图片的top5
                    # print(name)
                    if not os.path.exists("datasets/data/compitition/TestSetA/"+name):
                        name=temp+".jpeg"
                    if not os.path.exists("datasets/data/compitition/TestSetA/"+name):
                        name=temp+".png"
                        print(name)
                        print("??????")
                    save_file.write(name + ' ' +' '.join([str(p) for p in top_5]) + '\n')
                    count+=1
                continue
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths,
                                                   args)
                correct += acc1
                n += num_total
            else:
                # print("!!!!!!!!!")
                # print("y",y[0],'\n',"y.shape",y.shape)
                # print("X",'x','\n',"x.shape",x.shape)
                # assert False 
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data[
                    'metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)
        

        if args.current_epoch==-1:# 比赛测评写入文件

            print("写入完成,共计",count)
            save_file.close()


            # 压缩结果文件
            zip_file_path = './result.zip'
            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                zipf.write(save_path, os.path.basename(save_path))

            # 删除原文件
            # os.remove(save_path)
            print(f"{save_path} 已压缩为 {zip_file_path} 并删除原文件。")
            return
        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics


def eval_single_batch_dataset(image_classifier, dataset, args,
                              classification_head, data):

    model = image_classifier
    input_key = 'images'

    model.eval()
    classification_head.eval()

    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n, cnt_loss = 0., 0., 0., 0.

        data = maybe_dictionarize(data)
        x = data[input_key].to(device)
        y = data['labels'].to(device)

        assert x.shape[0] == 2 * args.k, 'val mismatch size'

        if 'image_paths' in data:
            image_paths = data['image_paths']

        logits = utils.get_logits(x, model, classification_head)

        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, device)

        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)

        cnt_loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        if hasattr(dataset, 'accuracy'):
            acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
            correct += acc1
            n += num_total
        else:
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels.append(y.cpu().clone().detach())
            all_preds.append(logits.cpu().clone().detach())
            metadata = data['metadata'] if 'metadata' in data else image_paths
            all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics['top1'], cnt_loss.item()


def evaluate(image_classifier,
             args,
             classification_head,
             train_stats={},
             logger=None):
    if args.eval_datasets is None:
        return
    info = vars(args)
    if args.current_epoch==-1:
        #用于 比赛提交的分支,用epoch=-1来区分
        dataset_class = getattr(datasets,"CompititionUpload")
        dataset = dataset_class(image_classifier.module.val_preprocess,
                                    location=args.data_location,
                                    batch_size=args.batch_size)
        eval_single_dataset(image_classifier, dataset, args, classification_head)
        return
    else:
        for i, dataset_name in enumerate(args.eval_datasets):
            print('Evaluating on', dataset_name)
            dataset_class = getattr(datasets, dataset_name)
            dataset = dataset_class(image_classifier.module.val_preprocess,
                                    location=args.data_location,
                                    batch_size=args.batch_size)

            results = eval_single_dataset(image_classifier, dataset, args,
                                        classification_head)

            if 'top1' in results:
                print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
                if logger != None:
                    logger.info(
                        f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
                train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)

            for key, val in results.items():
                if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                    print(f"{dataset_name} {key}: {val:.4f}")
                    if logger != None:
                        logger.info(f"{dataset_name} {key}: {val:.4f}")
                    train_stats[dataset_name + key] = round(val, 4)

        return info