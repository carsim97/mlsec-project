import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from robustbench.utils import load_model

from secml.array import CArray
from secml.figure import CFigure


def load_models(names, model_type):
    models = []
    if model_type == 'surr':
        output_dir = './models/surrogate'
    else:
        output_dir = './models/target'
    for i, name in enumerate(names):
        models.append(load_model(model_name=name, norm='Linf', model_dir=output_dir))

    return models


# This attack is based on the PGD algorithm; the only difference is that the gradient is computed using the sum of each
# loss related to each surrogate model.
# In this way, it is possible to fool multiple models at the same time with a single perturbed sample.
def run_attack(ds, epsilons, alpha, steps, n_adv):
    x_true = []
    y_true = []
    x_advs = []

    sample_idx = 0
    for x, y in ds:
        sample_idx += 1
        x = x.view((1, *x.shape))
        x_adv = x.clone()
        x_adv = x_adv.requires_grad_()

        loss = torch.nn.CrossEntropyLoss()

        invalid = False

        for i in range(steps):
            scores = []
            total_loss = 0
            for clf in surr_models:
                sc = clf(x_adv)

                # If clean sample is misclassified by any of the classifiers, then discard it
                if i == 0 and sc.argmax(dim=-1) != y:
                    invalid = True
                    print("Sample {} of {}: discarded".format(sample_idx, len(ds)))
                    break

                scores.append(sc)
                # Sum losses computed by each model
                loss_val = loss(sc, torch.LongTensor([y]))
                total_loss += loss_val

            if invalid:
                break

            # Compute total loss gradient
            total_loss.backward()
            gradient = x_adv.grad

            # Normalize the gradient (takes only the direction and discards the magnitude)
            # avoid division by 0
            if torch.norm(gradient, p=np.inf) != 0:
                gradient = gradient / torch.norm(gradient, p=np.inf)

            # Make step
            x_adv.data = x_adv.data + alpha * gradient
            x_adv.data = torch.clamp(x_adv, 0, 1)

            # Project inside epsilon-ball
            if torch.norm(x_adv - x, p=np.inf) > epsilons:
                delta = x_adv.data - x.data
                delta = delta / torch.norm(delta, p=np.inf)
                x_adv.data = x.data + epsilons * delta.data

            x_adv.grad.data.zero_()

        # If advx does not bypass all the surrogate models at the same time, then discard it
        for sc in scores:
            if sc.argmax(dim=-1) == y:
                invalid = True
                print("Sample {} of {}: discarded".format(sample_idx, len(ds)))
                break

        if not invalid:
            print("Sample {} of {}: added".format(sample_idx, len(ds)))
            x_true.append(x)
            y_true.append(y)
            x_advs.append(x_adv)

        # If desired number of advx to create is reached, stop advx generation
        if len(x_advs) == n_adv:
            print("{} adversarial examples created!".format(n_adv))
            break

        # print(f'Original label: {dataset.classes[y]}')
        # print(f'Adv loss: {total_loss}')
        # for i, scores_i in enumerate(scores):
        #     print(f'Adv label by clf {i + 1}: {dataset.classes[scores_i.argmax(dim=-1)]}')

    return x_true, y_true, x_advs


def compare_images(x, y, x_adv, models, image_index):
    sample = x[image_index]
    label = y[image_index]
    adv_sample = x_adv[image_index]
    adv_title = "Adversarial:"
    for i, clf in enumerate(models):
        output = clf(adv_sample)
        adv_label = output.argmax(dim=-1)
        adv_title += "\nModel {}: {}".format(i+1, dataset.classes[adv_label])

    f, ax = plt.subplots(1, 3)
    orig_img = torch.squeeze(sample, 0).permute(1, 2, 0)
    adv_img = torch.squeeze(adv_sample.detach(), 0).permute(1, 2, 0)
    diff_img = adv_img - orig_img
    diff_img -= diff_img.min()
    diff_img /= diff_img.max()

    ax[0].imshow(orig_img)
    ax[0].set_title("Original: {}".format(dataset.classes[label]))
    ax[1].imshow(adv_img)
    ax[1].set_title(adv_title)
    ax[2].imshow(diff_img)
    ax[2].set_title("Amplified perturbation")
    plt.show()


def eval_transferability(samples, labels, adv_samples, models):
    origin_error = []
    transfer_error = []

    for i, clf in enumerate(models):
        origin_miss = 0
        transfer_miss = 0
        for x, y, x_adv in zip(samples, labels, adv_samples):
            y_pred = clf(x).argmax(dim=-1).item()
            origin_miss += (y != y_pred)

            y_pred = clf(x_adv).argmax(dim=-1).item()
            transfer_miss += (y != y_pred)

        origin_error_clf = origin_miss / len(samples)
        origin_error.append(origin_error_clf)
        trans_error_clf = transfer_miss / len(samples)
        transfer_error.append(trans_error_clf)

    origin_acc = CArray(origin_error) * 100
    transfer_acc = CArray(transfer_error) * 100

    return origin_acc, transfer_acc


if __name__ == "__main__":
    # Load surrogate models
    surr_model_names = ['Chen2020Efficient',
                        'Sridhar2021Robust_34_15',
                        'Huang2021Exploring_ema']
    surr_models = load_models(surr_model_names, model_type='surr')

    # Load CIFAR10 dataset
    dataset = datasets.CIFAR10(root='.', download=True, train=False, transform=transforms.ToTensor())

    # Attack parameters
    iterations = 50
    eps = 0.25
    step_size = 0.1*eps
    n_advx = 6

    # Run attack
    x0, y0, advx = run_attack(dataset, epsilons=eps, alpha=step_size, steps=iterations, n_adv=n_advx)

    # img_idx = 0
    # compare_images(x0, y0, advx, surr_models, img_idx)

    # Load target models
    targ_model_names = ['Sehwag2021Proxy_ResNest152',
                        'Gowal2020Uncovering_28_10_extra',
                        'Carmon2019Unlabeled',
                        'Gowal2021Improving_R18_ddpm_100m',
                        'Addepalli2022Efficient_WRN_34_10',
                        'Chen2021LTD_WRN34_10',
                        'Sehwag2021Proxy_R18']
    targ_models = load_models(targ_model_names, model_type='targ')

    # Test adversarial examples transferability to target models
    orig_acc, trans_acc = eval_transferability(x0, y0, advx, targ_models)

    # Display results
    fig = CFigure(height=1)
    a = fig.sp.imshow(orig_acc.reshape((1, len(targ_models))),
                      cmap='Oranges', interpolation='nearest',
                      alpha=.65, vmin=60, vmax=70)

    fig.sp.xticks(CArray.arange(len(targ_models)))
    fig.sp.xticklabels(targ_model_names,
                       rotation=45, ha="right", rotation_mode="anchor")
    fig.sp.yticks([])

    for model_idx in range(len(targ_models)):
        fig.sp.text(model_idx, 0, orig_acc[model_idx].round(2).item(), va='center', ha='center')

    fig.sp.title("Test error of target classifiers (no attack) (%)")

    fig = CFigure(height=1)
    a = fig.sp.imshow(trans_acc.reshape((1, len(targ_models))),
                      cmap='Oranges', interpolation='nearest',
                      alpha=.65, vmin=60, vmax=70)

    fig.sp.xticks(CArray.arange(len(targ_models)))
    fig.sp.xticklabels(targ_model_names,
                       rotation=45, ha="right", rotation_mode="anchor")
    fig.sp.yticks([])

    for model_idx in range(len(targ_models)):
        fig.sp.text(model_idx, 0, trans_acc[model_idx].round(2).item(), va='center', ha='center')

    fig.sp.title("Test error of target classifiers under attack (%)")

    fig.show()
