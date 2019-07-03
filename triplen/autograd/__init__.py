from .function import AccumulateGrad


def backward(tensor, gradient=None):
    if not tensor.requires_grad:
        return
    backward_stack = [(tensor.grad_fn, gradient)]
    while len(backward_stack) != 0:
        grad_fn, gradient = backward_stack.pop()
        if grad_fn is None:
            continue
        outputs = grad_fn.apply(gradient)
        if isinstance(grad_fn, AccumulateGrad):
            continue
        # print(sys.getrefcount(grad_fn))
        if isinstance(outputs, tuple):
            assert len(outputs) == len(grad_fn.next_functions)
            for func, grad in zip(grad_fn.next_functions, outputs):
                backward_stack.append((func, grad))
        else:
            # assert len([func for func in grad_fn.next_functions if func is not None]) == 1
            for func in grad_fn.next_functions:
                backward_stack.append((func, outputs))
