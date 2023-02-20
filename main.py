import numpy as np
from secml.array import CArray
from secml.data import CDataset
from secml.data.loader.c_dataloader_cifar import CDataLoaderCIFAR10
from secml.ml.features.normalization import CNormalizerMinMax
from robustbench.utils import load_model
from secml.ml.classifiers import CClassifierPyTorch

from secml.adv.attacks.evasion import CFoolboxPGDLinf
from secml.ml.peval.metrics import CMetricTestError

from secml.figure import CFigure
from math import ceil


# Load models (surrogate or target)
def load_models(names, model_type):
    models = []
    if model_type == 'surr':
        output_dir = './models/surrogate'
    else:
        output_dir = './models/target'
    for i, name in enumerate(names):
        model = load_model(model_name=name, norm='Linf', model_dir=output_dir)
        models.append(CClassifierPyTorch(model, input_shape=(3, 32, 32), pretrained=True))

    return models


# Load CIFAR10 dataset
def load_cifar10():
    tr, ts = CDataLoaderCIFAR10().load()

    # Select test set and normalize
    x, y = ts.X, ts.Y
    normalizer = CNormalizerMinMax().fit(tr.X)
    x = normalizer.transform(x)

    return x, y


# Assess surrogate models' accuracy against clean samples,
# then discard clean samples that are misclassified by any of the surrogate models
def discard_miscl(x, y, models):
    for i, model in enumerate(models):
        y_pred = models[i].predict(x)
        idx = [(y[i] == y_pred[i]).item() for i in range(y_pred.size)]
        x = x[idx, :]
        y = y[idx]

    return x, y


# Craft adversarial examples based on PGD attack (using the CFoolboxPGDLinf class)
# Adversarial examples are crafted by taking into consideration one surrogate model at a time, to perform PGD attack
# on the list of test samples correctly classified by all the models.
# This will result in crafting 3 adversarial examples per test sample.
def run_attack(x, y, models):
    # PGD attack parameters
    y_target = None  # untargeted attack
    eps = 0.01
    alpha = 0.05
    steps = 50

    # Start attack
    adv_ds = CDataset(CArray.zeros((0, x.shape[1])), CArray.zeros((0,)))
    for i, model in enumerate(models):
        adv_ds_tmp = CDataset(CArray.zeros((x.shape[0], x.shape[1])), CArray.zeros((y.shape[0],)))
        y_pred = CArray.zeros((y.shape[0],))

        pgd_attack = CFoolboxPGDLinf(
            model, y_target=y_target,
            lb=0, ub=1,
            epsilons=eps,
            rel_stepsize=alpha,
            steps=steps,
            random_start=False
        )

        for j in range(x.shape[0]):
            print('Sample ' + str(j+1) + ' of ' + str(x.shape[0]))
            y_pred_el, _, adv_ds_el, _ = pgd_attack.run(x[j, :], y[j])
            y_pred[j] = y_pred_el
            adv_ds_tmp[j, :] = adv_ds_el[0, :]

            # adv_X = adv_ds_tmp.X
            # adv_X.save('./adv_example_X_all', overwrite=True)

        # # Assess surrogate models' accuracy against adversarial examples
        # accuracy = metric.performance_score(y, y_pred)
        # print("Model" + str(i+1) + " Accuracy after attack: " + str(accuracy))

        adv_ds.append(adv_ds_tmp)

    x_tmp = CArray.concatenate(x, x, axis=0)
    x = CArray.concatenate(x_tmp, x, axis=0)
    y_tmp = CArray.concatenate(y, y)
    y = CArray.concatenate(y_tmp, y)

    return x, y, adv_ds.X


# Discard adversarial examples unable to fool all the surrogate models at the same time
def discard_ineffective_examples(x, y, x_adv, models):
    for model in models:
        y_pred = model.predict(x_adv)
        idx = [(y[i] != y_pred[i]).item() for i in range(y_pred.size)]
        x = x[idx, :]
        y = y[idx]
        x_adv = x_adv[idx, :]

    return x, y, x_adv


def transferability_eval(x, y, x_adv, models):
    origin_error = []
    trans_error = []
    metric = CMetricTestError()

    for i, model in enumerate(models):
        y_pred = model.predict(x)
        origin_error_clf = metric.performance_score(y_true=y, y_pred=y_pred)
        origin_error.append(origin_error_clf)

        y_pred = model.predict(x_adv)
        trans_error_clf = metric.performance_score(y_true=y, y_pred=y_pred)
        trans_error.append(trans_error_clf)

    origin_acc = CArray(origin_error) * 100
    trans_acc = CArray(trans_error) * 100

    return origin_acc, trans_acc,


def compare_images(x, y, x_adv, model, idx):
    dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    y_pred = model.predict(x_adv)

    img_normal = x[idx, :].tondarray().reshape((3, 32, 32)).transpose(2, 1, 0)
    img_adv = x_adv[idx, :].tondarray().reshape((3, 32, 32)).transpose(2, 1, 0)
    img_normal = np.rot90(img_normal, -1)
    img_adv = np.rot90(img_adv, -1)

    diff_img = img_normal - img_adv
    diff_img -= diff_img.min()
    diff_img /= diff_img.max()

    fig = CFigure()
    fig.subplot(1, 3, 1)
    fig.sp.imshow(img_normal)
    fig.sp.title('{0}'.format(dataset_labels[int(y[idx].item())]))
    fig.sp.xticks([])
    fig.sp.yticks([])

    fig.subplot(1, 3, 2)
    fig.sp.imshow(img_adv)
    fig.sp.title('{0}'.format(dataset_labels[int(y_pred[idx].item())]))
    fig.sp.xticks([])
    fig.sp.yticks([])

    fig.subplot(1, 3, 3)
    fig.sp.imshow(diff_img)
    fig.sp.title('Amplified perturbation')
    fig.sp.xticks([])
    fig.sp.yticks([])
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    # Load surrogate models
    surr_model_names = ['Chen2020Efficient', 'Sridhar2021Robust_34_15', 'Huang2021Exploring_ema']
    surr_models = load_models(surr_model_names, model_type='surr')

    # # Load CIFAR10 test set
    # x0, y0 = load_cifar10()
    #
    # # Keep only correctly classified samples
    # x0, y0 = discard_miscl(x0, y0, surr_models)
    # print("N. of adversarial examples to generate: {}".format(x0.shape[0]))

    # # TO SKIP PREVIOUS COMPUTATION, uncomment the following lines ---------------------
    # x0 = CArray.load('./true_X_all')
    # y0 = CArray.load('./true_Y_all').flatten()
    # # ---------------------------------------------------------------------------------

    # # Generate adversarial examples
    # x0, y0, adv_X = run_attack(x0, y0, surr_models)
    #
    # # Save clean and perturbed samples
    # x0.save('./true_X_all')
    # y0.save('./true_Y_all')
    # adv_X.save('./adv_example_X_all')

    # # TO SKIP PREVIOUS COMPUTATION. uncomment the following lines ---------------------
    # x0 = CArray.load('./true_X_all')
    # y0 = CArray.load('./true_Y_all').flatten()
    # adv_X = CArray.load('./adv_example_X_all')
    # # ---------------------------------------------------------------------------------

    # # Discard adversarial examples unable to fool all the surrogate models at the same time
    # x0, y0, adv_X = discard_ineffective_examples(x0, y0, adv_X, surr_models)
    # print("N. of adversarial examples effective against the surrogate models: {}".format(adv_X.shape[0]))

    # # Save effective adversarial examples
    # x0.save('./true_X')
    # y0.save('./true_Y')
    # adv_X.save('./adv_example_X')

    # TO SKIP PREVIOUS COMPUTATION. uncomment the following lines ---------------------
    x0 = CArray.load('./true_X')
    y0 = CArray.load('./true_Y').flatten()
    adv_X = CArray.load('./adv_example_X')
    # ---------------------------------------------------------------------------------

    # # For the selected sample, display clear image, perturbed image and amplified perturbation
    # img_idx = 4
    # compare_images(x0, y0, adv_X, surr_models[0], img_idx)

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
    orig_acc, transf_acc = transferability_eval(x0, y0, adv_X, targ_models)

    fig = CFigure(height=1)
    a = fig.sp.imshow(orig_acc.reshape((1, len(targ_models))),
                      cmap='Oranges', interpolation='nearest',
                      alpha=.65, vmin=60, vmax=70)

    fig.sp.xticks(CArray.arange(len(targ_models)))
    fig.sp.xticklabels(targ_model_names,
                       rotation=45, ha="right", rotation_mode="anchor")
    fig.sp.yticks([])

    for i in range(len(targ_models)):
        fig.sp.text(i, 0, orig_acc[i].round(2).item(), va='center', ha='center')

    fig.sp.title("Test error of target classifiers (no attack) (%)")

    fig = CFigure(height=1)
    a = fig.sp.imshow(transf_acc.reshape((1, len(targ_models))),
                      cmap='Oranges', interpolation='nearest',
                      alpha=.65, vmin=60, vmax=70)

    fig.sp.xticks(CArray.arange(len(targ_models)))
    fig.sp.xticklabels(targ_model_names,
                       rotation=45, ha="right", rotation_mode="anchor")
    fig.sp.yticks([])

    for i in range(len(targ_models)):
        fig.sp.text(i, 0, transf_acc[i].round(2).item(), va='center', ha='center')

    fig.sp.title("Test error of target classifiers under attack (%)")

    fig.show()
