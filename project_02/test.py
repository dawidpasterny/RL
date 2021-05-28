import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
from utils import get_mnist_data
from models import ConvNN
from training_and_evaluation import train_model, predict_model
from attacks import fast_gradient_attack
from torch.nn.functional import cross_entropy

from math import ceil
from typing import Callable, Union, Tuple, List, Dict

from scipy.stats import norm, binom_test
from torch import nn
from statsmodels.stats.proportion import proportion_confint

from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm

    
class ConvNN(nn.Module):
    """
    A simple convolutional neural network for image classification on MNIST.
    """

    def __init__(self):
        super(ConvNN, self).__init__()
        self.sequential = nn.Sequential(
                             nn.Conv2d(1, 5, 5),
                             nn.ReLU(),
                             nn.BatchNorm2d(5),
                             nn.MaxPool2d(2),
                             nn.Conv2d(5, 5, 5),
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             nn.Flatten(),
                             nn.Linear(80, 10),
                            )

    def forward(self, input):
        assert input.min() >= 0 and input.max() <= 1.
        return self.sequential(input)

    def device(self):
        """
        Convenience function returning the device the model is located on.
        """
        return next(self.parameters()).device


def lower_confidence_bound(num_class_A: int, num_samples: int, alpha: float) -> float:
    """
    Computes a lower bound on the probability of the event occuring in a Bernoulli distribution.
    Parameters
    ----------
    num_class_A: int
        The number of times the event occured in the samples.
    num_samples: int
        The total number of samples from the bernoulli distribution.
    alpha: float
        The desired confidence level, e.g. 0.05.

    Returns
    -------
    lower_bound: float
        The lower bound on the probability of the event occuring in a Bernoulli distribution.

    """
    return proportion_confint(num_class_A, num_samples, alpha=2 * alpha, method="beta")[0]


class SmoothClassifier(nn.Module):
    """
    Randomized smoothing classifier.
    """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: nn.Module, num_classes: int, sigma: float):
        """
        Constructor for SmoothClassifier.
        Parameters
        ----------
        base_classifier: nn.Module
            The base classifier (i.e. f(x)) that maps an input sample to a logit vector.
        num_classes: int
            The number of classes.
        sigma: float
            The variance used for the Gaussian perturbations.
        """
        super(SmoothClassifier, self).__init__()
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def device(self):
        return self.base_classifier.device()

    def certify(self, inputs: torch.Tensor, n0: int, num_samples: int, alpha: float, batch_size: int) -> Tuple[int,
                                                                                                               float]:
        """
        Certify the input sample using randomized smoothing.

        Uses lower_confidence_bound to get a lower bound on p_A, the probability of the top class.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N is the image width/height.
            The input image to certify.
        n0: int
            Number of samples to determine the most likely class.
        num_samples: int
            Number of samples to use for the robustness certification.
        alpha: float
            The confidence level, e.g. 0.05 for an expected error rate of 5%.
        batch_size: int
           The batch size to use during the certification, i.e. how many noise samples to classify in parallel.

        Returns
        -------
        Tuple containing:
            * top_class: int. The predicted class g(x) of the input sample x. Returns -1 in case the classifier abstains
                         because the desired confidence level could not be reached.
            * radius: float. The radius for which the prediction can be certified. Is zero in case the classifier
                      abstains.

        """
        self.base_classifier.eval()

        ##########################################################
        # YOUR CODE HERE
        # Inference sampling
        class_counts = self._sample_noise_predictions(inputs, n0, batch_size)
        c = torch.argmax(class_counts)

        # Cartification sampling
        class_counts_cert = self._sample_noise_predictions(inputs, num_samples, batch_size)
        p_A_lower_bound = lower_confidence_bound(class_counts_cert[c], num_samples, alpha)
        ##########################################################

        if p_A_lower_bound < 0.5: # prediction doesn't have sufficient precedence over ALL others
            return SmoothClassifier.ABSTAIN, 0.0
        else:
            ##########################################################
            # YOUR CODE HERE
            top_class = c
            radius = self.sigma*norm.ppf(p_A_lower_bound)
            ##########################################################
            return top_class, radius


    def predict(self, inputs: torch.tensor, num_samples: int, alpha: float, batch_size: int) -> int:
        """
        Predict a label for the input sample via the smooth classifier g(x).

        Uses the test binom_test(count1, count1+count2, p=0.5) > alpha to determine whether the top class is the winning
        class with at least the confidence level alpha.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N is the image width/height.
            The input image to predict.
        num_samples: int
            The number of samples to draw in order to determine the most likely class.
        alpha: float
            The desired confidence level that the top class is indeed the most likely class. E.g. alpha=0.05 means that
            the expected error rate must not be larger than 5%.
        batch_size: int
            The batch size to use during the prediction, i.e. how many noise samples to classify in parallel.

        Returns
        -------
        int: the winning class or -1 in case the desired confidence level could not be reached.
        """
        self.base_classifier.eval()
        class_counts = self._sample_noise_predictions(inputs, num_samples, batch_size).cpu()
        ##########################################################
        # YOUR CODE HERE
        sorted_idxs = class_counts.argsort(descending=True)
        c = sorted_idxs[0]
        c_score = class_counts[c]
        # with (1-alpha) confidence p_c lies above p_c_lb
        p_c_lb = lower_confidence_bound(class_counts[c], num_samples, alpha)

        scores = class_counts/num_samples
        # winning_class = c if all(p_c_lb>scores[torch.arange(self.num_classes)!=c]) else -1
        winning_class = c if binom_test(c_score, c_score+class_counts[sorted_idxs[1]], p=.5) else -1

        return winning_class
        ##########################################################


    def _sample_noise_predictions(self, inputs: torch.tensor, num_samples: int, batch_size: int) -> torch.Tensor:
        """
        Sample random noise perturbations for the input sample and count the predicted classes of the base classifier.

        Note: this function clamps the distorted samples in the valid range, i.e. [0,1].

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N is the image width/height.
            The input image to predict.
        num_samples: int
            The number of samples to draw.
        batch_size: int
            The batch size to use during the prediction, i.e. how many noise samples to classify in parallel.

        Returns
        -------
        torch.Tensor of shape [K,], where K is the number of classes.
        Each entry of the tensor contains the number of times the base classifier predicted the corresponding class for
        the noise samples.
        """
        num_remaining = num_samples
        with torch.no_grad():
            classes = torch.arange(self.num_classes).to(self.device())
            class_counts = torch.zeros([self.num_classes], dtype=torch.long, device=self.device())
            for it in range(ceil(num_samples / batch_size)):
                this_batch_size = min(num_remaining, batch_size)
                ##########################################################
                # YOUR CODE HERE
                # Assemble a batch for inference by distorting input a number of times
                x = torch.normal(torch.zeros(this_batch_size, *tuple(inputs[0].shape)), self.sigma)
                x += inputs
                scores = self.base_classifier(torch.clip(x,0,1))

                # Append class counts
                clss, counts = torch.unique(torch.max(scores,dim=1).indices, return_counts=True)
                class_counts[clss] += counts

                num_remaining -= this_batch_size
                ##########################################################
        return class_counts

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make a single prediction for the input batch using the base classifier and random Gaussian noise.

        Note: this function clamps the distorted samples in the valid range, i.e. [0,1].
        Parameters
        ----------
        inputs: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the number of channels,
               and N is the image width/height.
            The input batch of images to predict.
        Returns
        -------
        torch.Tensor of shape [B, K]
        The logits for each input image.
        """
        noise = torch.randn_like(inputs) * self.sigma
        return self.base_classifier((inputs + noise).clamp(0, 1))


def train_model(model: nn.Module, dataset: Dataset, batch_size: int, loss_function: Callable, optimizer: Optimizer,
                epochs: int = 1, loss_args: Union[dict, None] = None) -> Tuple[List, List]:
    """
    Train a model on the input dataset.
    Parameters
    ----------
    model: nn.Module
        The input model to be trained.
    dataset: torch.utils.data.Dataset
        The dataset to train on.
    batch_size: int
        The training batch size.
    loss_function: function with signature: (x, y, model, **kwargs) -> (loss, logits).
        The function used to compute the loss.
    optimizer: Optimizer
        The model's optimizer.
    epochs: int
        Number of epochs to train for. Default: 1.
    loss_args: dict or None
        Additional arguments to be passed to the loss function.

    Returns
    -------
    Tuple containing
        * losses: List[float]. The losses obtained at each step.
        * accuracies: List[float]. The accuracies obtained at each step.

    """
    if loss_args is None:
        loss_args = {}
    losses = []
    accuracies = []
    num_train_batches = int(torch.ceil(torch.tensor(len(dataset) / batch_size)).item())
    for epoch in range(epochs):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for x,y in tqdm(train_loader, total=num_train_batches):
            ##########################################################
            # YOUR CODE HERE
            optimizer.zero_grad()
            loss, logits = loss_function(x, y, model, **loss_args)
            accuracy = torch.count_nonzero(torch.max(logits, dim=1).indices==y)
            # accuracy = torch.mean(nn.functional.softmax(logits, dim=1)[:,y])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracies.append(accuracy.item()/batch_size)
            ##########################################################
    return losses, accuracies

def predict_model(model: nn.Module, dataset: Dataset, batch_size: int, attack_function: Union[Callable, None] = None,
                  attack_args: Union[Callable, None] = None) -> float:
    """
    Use the model to predict a label for each sample in the provided dataset. Optionally performs an attack via
    the attack function first.
    Parameters
    ----------
    model: nn.Module
        The input model to be used.
    dataset: torch.utils.data.Dataset
        The dataset to predict for.
    batch_size: int
        The batch size.
    attack_function: function or None
        If not None, call the function to obtain a perturbed batch before evaluating the prediction.
    attack_args: dict or None
        Additionall arguments to be passed to the attack function.

    Returns
    -------
    float: the accuracy on the provided dataset.
    """
    model.eval()
    if attack_args is None:
        attack_args = {}
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = int(torch.ceil(torch.tensor(len(dataset) / batch_size)).item())
    predictions = []
    targets = []
    for x, y in tqdm(test_loader, total=num_batches):
        ##########################################################
        # YOUR CODE HERE
        logits = model(x)
        if attack_function!=None:
            x_pert = attack_function(logits, x, y, **attack_args)
            logits = model(x_pert)
        predictions.append(torch.max(logits, dim=1).indices)
        targets.append(y)
        ##########################################################
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = (predictions == targets).float().mean().item()
    return accuracy

def fast_gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, epsilon: float, norm: str = "2",
                         loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack on the input x.
    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is the number of classes.
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the number of channels, and N is the image
       dimension.
       The input batch of images. Note that x.requires_grad must have been active before computing the logits
       (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation (before clipping) will have a norm of
        exactly epsilon as measured by the desired norm (see argument: norm).
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", the perturbation (before clipping)
         will have a L_1 norm of exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.

    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    ##########################################################
    # YOUR CODE HERE
    # x.grad = None
    loss = loss_fn(logits, y)
    loss.backward(retain_graph=True)
    # loss.backward()

    # step in the direction of steepest ascent in image space but 
    # stay within a bal;l of radius epsilon measured in norm
    if norm=="inf":
        x_pert = x + epsilon*torch.sign(x.grad) 
    else:
        x_pert = x + epsilon*x.grad/torch.norm(x.grad, p=float(norm), dim=(1, 2, 3), keepdim=True)

    x_pert = torch.clip(x_pert, 0.0, 1.0) # clip pixel values
    ##########################################################

    return x_pert.detach()




if __name__ == "__main__":
    mnist_trainset = get_mnist_data(train=True)
    mnist_testset = get_mnist_data(train=False)

    use_cuda = torch.cuda.is_available() #and False

    model = ConvNN()
    if use_cuda:
        model = model.cuda()
    
    # print(list(model.parameters()))
    epochs = 1
    batch_size = 128
    test_batch_size = 1000  # feel free to change this
    lr = 1e-3

    opt = Adam(model.parameters(), lr=lr)

    def loss_function(x, y, model):
        logits = model(x).cpu()
        loss = cross_entropy(logits, y)
        return loss, logits

    # losses, accuracies = train_model(model, mnist_trainset, batch_size=batch_size, loss_function=loss_function, optimizer=opt)
    # plt.plot(losses)
    # plt.show()
    model.load_state_dict(torch.load("models/standard_training.checkpoint", map_location="cpu"))

    # # Perturbation
    test_loader = DataLoader(mnist_testset, batch_size = 10, shuffle=True)
    x,y = next(iter(test_loader))

    # attack_args_l2 = {"epsilon": 5, "norm": "2"}
    # attack_args_linf = {"epsilon": 0.5, "norm": "inf"}

    # logits = model(x.requires_grad_(True))
    # x_pert_linf = fast_gradient_attack(logits, x, y, **attack_args_linf)
    # y_pert_linf = torch.max(model(x_pert_linf), dim=1).indices
    # x_pert_l2 = fast_gradient_attack(logits, x, y, **attack_args_l2)
    # y_pert_l2 = torch.max(model(x_pert_l2), dim=1).indices

    # for ix in range(len(x)):
    #     plt.subplot(131)
    #     plt.imshow(x[ix,0].detach().cpu(), cmap="gray")
    #     plt.title(f"Label: {y[ix]}")
    #     plt.subplot(132)
    #     plt.imshow(x_pert_l2[ix,0].detach().cpu(), cmap="gray")
    #     plt.title(f"Predicted: {y_pert_l2[ix]}")
    #     plt.subplot(133)
    #     plt.imshow(x_pert_linf[ix,0].detach().cpu(), cmap="gray")
    #     plt.title(f"Predicted: {y_pert_linf[ix]}")
    #     plt.show()

    # Adverserial training
    attack_args = {'norm': "2", "epsilon": 5}
    
    def loss_function(x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,  **attack_args) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loss function used for adversarial training. First computes adversarial examples on the input batch via fast_gradient_attack and then computes the logits
        and the loss on the adversarial examples.
        Parameters
        ----------
        x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the number of channels, and N is the image width/height.
            The input batch to certify.
        y: torch.Tensor of shape [B, 1].
            The labels of the input batch.
        model: torch.nn.Module
            The classifier to be evaluated.
        attack_args: additional arguments passed to the adversarial attack function.
        
        Returns
        -------
        Tuple containing
            * loss_pert: torch.Tensor, shape [B,]
                The loss obtained on the adversarial examples.
            * logits_pert: torch.Tensor, shape [B, K], where K is the number of classes.
                The logits obtained on the adversarial examples
        """
        ##########################################################
        # YOUR CODE HERE
        logits = model(x.requires_grad_(True))
        x_pert = fast_gradient_attack(logits, x, y, **attack_args)
        model.zero_grad()

        logits_pert = model(x_pert)
        loss_pert = cross_entropy(logits_pert, y)
        ##########################################################
        # Important: don't forget to call model.zero_grad() after creating the adversarial examples.
        return loss_pert, logits_pert

    # losses, accuracies = train_model(model, mnist_trainset, batch_size=batch_size, loss_function=loss_function, optimizer=opt, loss_args=attack_args, epochs=epochs)
    # plt.plot(losses)
    # torch.save(model.state_dict(), "models/adversarial_training.checkpoint")
    # model.load_state_dict(torch.load("models/adversarial_training.checkpoint", map_location="cpu"))
    
    sigma = 1
    batch_size = 128
    lr = 1e-3
    epochs = 1
    model = SmoothClassifier(base_classifier=ConvNN(), num_classes=10, sigma=sigma)
    print(f"Label: {y[0]}")
    top_class, radius = model.certify([x[0]], 25, 250, .1, 50)
    print(f"Top class: {top_class}")
    print(f"Radius: {radius}")