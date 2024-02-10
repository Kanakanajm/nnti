import activations
import layers


class Model:
    def __init__(self, components) -> None:
        """
        expects a list of components of the model in order with which they must be applied
        """
        self.components = components

    def forward(self, x):
        """
        performs forward pass on the input x using all components from self.components
        """
        y = x.copy()
        for component in self.components:
            y = component(y)
        return y

    def backward(self, in_grad):
        """
        expects in_grad - a gradient of the loss w.r.t. output of the model
        in_grad must be of the same size as the output of the model

        returns dictionary, where
            key - index of the component in the component list
            value - value of the gradient for that component
        """
        backward_ = {}
        backward_order = self.components.copy()
        backward_order.reverse()
        for i, component in enumerate(backward_order):
            result = component.grad(in_grad)
            if type(result) is tuple:  # Linear layer
                in_grad = result[1]
                backward_[len(self.components) - i - 1] = result[0]
            else:  # Activation functions
                in_grad = result

        return backward_

    def update_parameters(self, grads, lr):
        """
        performs one gradient step with learning rate lr for all components
        """
        for i, grad in grads.items():
            self.components[i].weights -= lr * grad
