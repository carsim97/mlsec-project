import numpy as np
from secml.array import CArray
from secml.data.loader.c_dataloader_cifar import CDataLoaderCIFAR10
from secml.ml.features.normalization import CNormalizerMinMax
from robustbench.utils import load_model
from secml.ml.classifiers import CClassifierPyTorch

from secml.ml.classifiers.loss import CLossCrossEntropy
from secml.ml.peval.metrics import CMetricTestError

from secml.figure import CFigure


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


# Gradient descent optimization algorithm
def gd_untargeted(start, label, model, alpha=0.1, max_iter=200):
    x = start.deepcopy()
    y = label
    loss_func = CLossCrossEntropy()

    for i in range(max_iter):
        scores = model.decision_function(x)

        # Gradient of the loss w.r.t. the clf logits
        loss_gradient = loss_func.dloss(y_true=y, score=scores)
        # Gradient of the clf logits w.r.t. the input
        clf_gradient = model.grad_f_x(x, y)

        # Chain rule
        gradient = clf_gradient * loss_gradient

        # Normalize the gradient (takes only the direction and discards the magnitude)
        # avoid division by 0
        if gradient.norm(np.inf) != 0:
            gradient /= gradient.norm(np.inf)

        # Make step
        x = x + alpha * gradient

        # Stop condition: the targeted point crosses the decision boundary (misclassification is achieved).
        # If condition is never met, the computation stops when i == max_iter.
        if model.predict(x) != y:
            break

    return x


# Generate universal adversarial perturbation through a gradient descent optimization approach.
# Such a perturbation can be applied to clean samples to fool all the surrogate models at the same time.
def run_attack(x, y, models, xi, min_advx_to_create):
    v = CArray.zeros((1, x.shape[1]))
    n_advx = 0
    fooling_rate = 0.0
    true_x = []
    true_y = []
    adv_x = []

    # UAP algorithm (Moosavi-Dezfooli, 2016)
    # Stop condition: meet a target accuracy (in this case, I want to create at least 100 adversarial examples)
    while fooling_rate <= min_advx_to_create/x.shape[0]:
        for i in range(x.shape[0]):
            print('Sample', str(i+1), 'of', str(x.shape[0]))
            for model in models:
                y_perturb = model.predict(x[i, :] + v)
                # If misclassification after adding perturbation is not achieved,
                # apply gradient descent to further perturb the sample.
                if y_perturb == y[i]:
                    x_perturb = gd_untargeted(x[i, :] + v, y[i], model)

                    # Project inside xi-ball
                    dv = x[i, :] + v - x_perturb
                    if (v + dv).norm(np.inf) > xi:
                        v = (v + dv) / (v + dv).norm(np.inf) * xi
                    else:
                        v = v + dv

        # Count adversarial examples able to fool all the surrogate models
        # at the same time.
        for i in range(x.shape[0]):
            count = 0
            for model in models:
                y_pred = model.predict(x[i, :] + v)
                if y[i] != y_pred:
                    count += 1
            if count == 3:
                n_advx += 1
                true_x.append(x[i, :].tolist())
                true_y.append(y[i].item())
                adv_x.append((x[i, :] + v).tolist())

        # Compute fooling rate (test error)
        fooling_rate = n_advx / x.shape[0]

    print('Fooling rate:', str(fooling_rate))

    # Return clean and perturbed adversarial examples
    return CArray(true_x), CArray(true_y), CArray(adv_x)


# Evaluate the transferability of adversarial examples to target models
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

    return origin_acc, trans_acc


def compare_images(x, y, x_adv, models, model_names, idx):
    dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    img_normal = x[idx, :].tondarray().reshape((3, 32, 32)).transpose(2, 1, 0)
    img_adv = x_adv[idx, :].tondarray().reshape((3, 32, 32)).transpose(2, 1, 0)
    img_normal = np.rot90(img_normal, -1)
    img_adv = np.rot90(img_adv, -1)

    diff_img = img_normal - img_adv
    diff_img -= diff_img.min()
    diff_img /= diff_img.max()

    f = CFigure()
    f.sp.imshow(diff_img)
    f.sp.title('Amplified perturbation')
    f.sp.xticks([])
    f.sp.yticks([])
    f.tight_layout()

    f1 = CFigure()
    f1.subplot(1, 4, 1)
    f1.sp.imshow(img_normal)
    f1.sp.title('Original: ' + dataset_labels[int(y[idx].item())])
    f1.sp.xticks([])
    f1.sp.yticks([])

    for i, model in enumerate(models):
        y_pred = model.predict(x_adv[idx, :])

        f1.subplot(1, 4, (i+2))
        f1.sp.imshow(img_adv)
        f1.sp.title(model_names[i] + ': ' + dataset_labels[int(y_pred.item())])
        f1.sp.xticks([])
        f1.sp.yticks([])

    f.show()
    f1.show()


if __name__ == '__main__':
    # Load surrogate models
    surr_model_names = ['Chen2020Efficient',
                        'Sridhar2021Robust_34_15',
                        'Huang2021Exploring_ema']
    surr_models = load_models(surr_model_names, model_type='surr')

    # Load CIFAR10 test set (10000 samples)
    x0, y0 = load_cifar10()
    x0 = x0[:, :]
    y0 = y0[:]

    # Keep only correctly classified samples (8178 samples)
    x0, y0 = discard_miscl(x0, y0, surr_models)
    x0 = x0[:, :]
    y0 = y0[:]
    print("N. of adversarial examples to generate: {}".format(x0.shape[0]))

    # Run UAP attack on 8178 samples (wanting to get at least 100 adversarial examples)
    xi = 0.01
    min_advx_to_create = 100
    x0, y0, adv_x = run_attack(x0, y0, surr_models, xi, min_advx_to_create)

#     # Display clear image, perturbed image and amplified perturbation of the selected sample
#     img_idx = 1
#     compare_images(x0, y0, adv_x, surr_models, surr_model_names, img_idx)

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
    orig_acc, transf_acc = transferability_eval(x0, y0, adv_x, targ_models)

    # Display transferability results
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
    a = fig.sp.imshow(transf_acc.reshape((1, len(targ_models))),
                      cmap='Oranges', interpolation='nearest',
                      alpha=.65, vmin=60, vmax=70)

    fig.sp.xticks(CArray.arange(len(targ_models)))
    fig.sp.xticklabels(targ_model_names,
                       rotation=45, ha="right", rotation_mode="anchor")
    fig.sp.yticks([])

    for model_idx in range(len(targ_models)):
        fig.sp.text(model_idx, 0, transf_acc[model_idx].round(2).item(), va='center', ha='center')

    fig.sp.title("Test error of target classifiers under attack (%)")

    fig.show()
