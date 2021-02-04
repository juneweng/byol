import paddle
import paddle.nn as nn
import random
import paddle.vision.transforms as T
import byol.transforms
# from paddle.vision.models import resnet50
import paddle.fluid as fluid
import copy
import paddle.nn.functional as F


__metaclass__ = type
from resnet import ResNet50

# helper functions


def default(val, def_val):
    return def_val if val is None else val

# def flatten(t):
#     return t.reshape(t.shape[0], -1)



# loss fn                                        

def loss_fn(x, y):                               
    x = F.normalize(x, axis=-1, p=2)              
    y = F.normalize(y, axis=-1, p=2)              
    return 2 - 2 * (x * y).sum(axis=-1)           

WRAPPER_ASSIGNMENTS = ('__module__', '__name__', '__qualname__', '__doc__',
                       '__annotations__')
WRAPPER_UPDATES = ('__dict__',)
################################################################################
### partial() argument application
################################################################################

# Purely functional, no descriptor behaviour

# def update_wrapper(wrapper,
#                    wrapped,
#                    assigned = WRAPPER_ASSIGNMENTS,
#                    updated = WRAPPER_UPDATES):
#     """Update a wrapper function to look like the wrapped function

#        wrapper is the function to be updated
#        wrapped is the original function
#        assigned is a tuple naming the attributes assigned directly
#        from the wrapped function to the wrapper function (defaults to
#        functools.WRAPPER_ASSIGNMENTS)
#        updated is a tuple naming the attributes of the wrapper that
#        are updated with the corresponding attribute from the wrapped
#        function (defaults to functools.WRAPPER_UPDATES)
#     """
#     for attr in assigned:
#         try:
#             value = getattr(wrapped, attr)
#         except AttributeError:
#             pass
#         else:
#             setattr(wrapper, attr, value)
#     for attr in updated:
#         getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
#     # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
#     # from the wrapped function when updating __dict__
#     wrapper.__wrapped__ = wrapped
#     # Return the wrapper so this can be used as a decorator via partial()
#     return wrapper
# def recursive_repr(fillvalue='...'):
#     'Decorator to make a repr function return fillvalue for a recursive call'

#     def decorating_function(user_function):
#         repr_running = set()

#         def wrapper(self):
#             key = id(self), get_ident()
#             if key in repr_running:
#                 return fillvalue
#             repr_running.add(key)
#             try:
#                 result = user_function(self)
#             finally:
#                 repr_running.discard(key)
#             return result

#         # Can't use functools.wraps() here because of bootstrap issues
#         wrapper.__module__ = getattr(user_function, '__module__')
#         wrapper.__doc__ = getattr(user_function, '__doc__')
#         wrapper.__name__ = getattr(user_function, '__name__')
#         wrapper.__qualname__ = getattr(user_function, '__qualname__')
#         wrapper.__annotations__ = getattr(user_function, '__annotations__', {})
#         return wrapper

#     return decorating_function



# class partial:
#     """New function with partial application of the given arguments
#     and keywords.
#     """

#     __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

#     def __new__(*args, **keywords):
#         if not args:
#             raise TypeError("descriptor '__new__' of partial needs an argument")
#         if len(args) < 2:
#             raise TypeError("type 'partial' takes at least one argument")
#         cls, func, *args = args
#         if not callable(func):
#             raise TypeError("the first argument must be callable")
#         args = tuple(args)

#         if hasattr(func, "func"):
#             args = func.args + args
#             tmpkw = func.keywords.copy()
#             tmpkw.update(keywords)
#             keywords = tmpkw
#             del tmpkw
#             func = func.func

#         self = super(partial, cls).__new__(cls)

#         self.func = func
#         self.args = args
#         self.keywords = keywords
#         return self

#     def __call__(*args, **keywords):
#         if not args:
#             raise TypeError("descriptor '__call__' of partial needs an argument")
#         self, *args = args
#         newkeywords = self.keywords.copy()
#         newkeywords.update(keywords)
#         return self.func(*self.args, *args, **newkeywords)

#     @recursive_repr()
#     def __repr__(self):
#         qualname = type(self).__qualname__
#         args = [repr(self.func)]
#         args.extend(repr(x) for x in self.args)
#         args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
#         if type(self).__module__ == "functools":
#             return f"functools.{qualname}({', '.join(args)})"
#         return f"{qualname}({', '.join(args)})"

#     def __reduce__(self):
#         return type(self), (self.func,), (self.func, self.args,
#                self.keywords or None, self.__dict__ or None)

#     def __setstate__(self, state):
#         if not isinstance(state, tuple):
#             raise TypeError("argument to __setstate__ must be a tuple")
#         if len(state) != 4:
#             raise TypeError(f"expected 4 items in state, got {len(state)}")
#         func, args, kwds, namespace = state
#         if (not callable(func) or not isinstance(args, tuple) or
#            (kwds is not None and not isinstance(kwds, dict)) or
#            (namespace is not None and not isinstance(namespace, dict))):
#             raise TypeError("invalid partial state")

#         args = tuple(args) # just in case it's a subclass
#         if kwds is None:
#             kwds = {}
#         elif type(kwds) is not dict: # XXX does it need to be *exactly* dict?
#             kwds = dict(kwds)
#         if namespace is None:
#             namespace = {}

#         self.__dict__ = namespace
#         self.func = func
#         self.args = args
#         self.keywords = kwds

# try:
#     from _functools import partial
# except ImportError:
#     pass
# def wraps(wrapped,
#           assigned = WRAPPER_ASSIGNMENTS,
#           updated = WRAPPER_UPDATES):
#     """Decorator factory to apply update_wrapper() to a wrapper function

#        Returns a decorator that invokes update_wrapper() with the decorated
#        function as the wrapper argument and the arguments to wraps() as the
#        remaining arguments. Default arguments are as for update_wrapper().
#        This is a convenience function to simplify applying partial() to
#        update_wrapper().
#     """
#     return partial(update_wrapper, wrapped=wrapped,
#                    assigned=assigned, updated=updated)


# def singleton(cache_key):
#     def inner_fn(fn):
#         @wraps(fn)
#         def wrapper(self, *args, **kwargs):
#             instance = getattr(self, cache_key)
#             if instance is not None:
#                 return instance

#             instance = fn(self, *args, **kwargs)
#             setattr(self, cache_key, instance)
#             return instance
#         return wrapper
#     return inner_fn


def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val



# augmentation utils

normalize = T.Normalize(mean=[0.485*255, 0.456*255, 0.406*255],
                        std=[0.229*255, 0.224*255, 0.225*255],
                        data_format='CHW'),     
augmentation = [
    byol.transforms.RandomApply([
        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
    ],p = 0.3),
    byol.transforms.RandomGrayscale(p=0.2),
    byol.transforms.RandomApply([byol.transforms.GaussianBlur((1.0, 2.0))],p=0.2),
    T.RandomResizedCrop(224, scale=(0.2, 1.)),
    # T.RandomResizedCrop((image_size, image_size)),
    normalize
]       


# MLP class for projector and predictor                                   
                                                                          
class MLP(nn.Layer):                                                     
    def __init__(self, dim, projection_size, hidden_size = 4096):         
        super(MLP, self).__init__()                                                
        self.net = nn.Sequential(                                         
            nn.Linear(dim, hidden_size),                                  
            nn.BatchNorm1D(hidden_size),                                  
            nn.ReLU(),                                        
            nn.Linear(hidden_size, projection_size)                       
        )                                                                 
                                                                          
    def forward(self, x):                                                 
        return self.net(x)                                                
                                                                          

class EMA():                                                       
    def __init__(self, beta):                                      
        # super(EMA).__init__()                                         
        self.beta = beta                                           
                                                                   
    def update_average(self, old, new):                            
        if old is None:                                            
            return new                                             
        return old * self.beta + (1 - self.beta) * new             

def update_moving_average(ema_updater, ma_model, current_model):                                    
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):        
        # old_weight, up_weight = ma_params.data, current_params.data 
        # tmp = current_params
        # current_params.set_value(ma_params)  
        # ma_params.set_value(tmp)                              
        ma_params.set_value(ema_updater.update_average(ma_params, current_params))


def init_online_encoder(ma_model, current_model):                                    
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):        
        # old_weight, up_weight = ma_params.data, current_params.data  
        # ma_params = current_params         
        ma_params.set_value(current_params)                      
        # ma_params.data = ema_updater.update_average(old_weight, up_weight)  
    # return ma_params                                                                                                   

# def init_online_encoder(ma_model, current_model):                                    
#     for cur_params in current_model.parameters():   
#         ma_params.data = cur_params.data     
#         # old_weight, up_weight = ma_params.data, current_params.data  
#         # current_params.data = ma_params.data 
#         return ma_params                               
#         # ma_params.data = ema_updater.update_average(old_weight, up_weight)      



# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Layer):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super(NetWrapper,self).__init__()
        self.net = net
        self.layer = layer

        # self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.projector = MLP(2048, self.projection_size, self.projection_hidden_size)
        self.hidden = None
        
                                                    
    def _find_layer(self):
        # print("here?")
        if type(self.layer) == str:

            modules = dict([self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            # print("self.net.named_modules(): ", str(self.net.children()))
            # print("self.layer: ", self.layer)
            children = list(self.net.children())
            return children[self.layer]
        return None

    # def _hook(self, _, __, output):
    #     self.hidden = flatten(output)

    # def _register_hook(self):
    #     layer = self._find_layer()
    #     # assert layer is not None, f'hidden layer ({self.layer}) not found'
    #     handle = layer.register_forward_hook(self._hook)
    #     self.hook_registered = True


    # @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        # print("dim", dim)
        projector = self.projector
        # print('projector',projector)
        # print('hidden', hidden)
        return projector

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)
        # if not self.hook_registered:
        #     layer = self._find_layer()
        #     self.hook_registered = True
        # else:
            # self._register_hook()
            # hidden = flatten(out)

        u = self.net(x)
        # print("u.shape",u.shape)
        # self.hidden = paddle.flatten(u,start_axis=-2, stop_axis=-1)
        # hidden = self.hidden
        hidden = u
        # print("hidden.shape",hidden.shape)
        _, dim = hidden.shape
        # o = self.MLPnet(hidden)
        # print("o.shape",o.shape)
        self.hidden = None
        # assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_embedding = False):
        representation = self.get_representation(x)
        # print("representation.shape",representation.shape)

        if return_embedding:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class BYOL(nn.Layer):
    def __init__(
        self,
        net,
        image_size=256,
        hidden_layer=-2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True
    ):
        super(BYOL, self).__init__()
        self.net = net

        self.augment1 = default(augment_fn, T.Compose(augmentation))
        self.augment2 = default(augment_fn2, self.augment1)
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)     
        self.target_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)     
        init_online_encoder(self.target_encoder,self.online_encoder)                                                                                        
        self.use_momentum = use_momentum                                                                       
        # self.target_encoder = None                                                                             
        self.target_ema_updater = EMA(moving_average_decay)                                                    
                                                                                                            
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)  

         # get device of network and make wrapper same device                        
        # device = get_module_device(net)                                             
        # self.to(device)                                                             
                                                                                    
        # send a mock image tensor to instantiate singleton parameters              
        # self.forward(fluid.layers.randn(2, 3, image_size, image_size, device=device))  
        # self.forward(paddle.randn(shape=[2, 3])) 


    # @singleton('target_encoder')                                                                                                                 
    def _get_target_encoder(self):                                                                                                               
        # target_encoder = copy.deepcopy(self.online_encoder) 
        target_encoder = self.target_encoder
        # target_encoder = self.online_encoder.clone()                                                                                 
        set_requires_grad(target_encoder, False)                                                                                                 
        return target_encoder                                                                                                                    
                                                                                                                                                
    def reset_moving_average(self):                                                                                                              
        del self.target_encoder                                                                                                                  
        self.target_encoder = None                                                                                                               
                                                                                                                                                
    def update_moving_average(self):                                                                                                             
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'      
        assert self.target_encoder is not None, 'target encoder has not been created yet'                                                        
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)                                                                                                                                                                   

    def forward(self, img_one, img_two):
        # if return_embedding:
        #     return self.online_encoder(x)

        # image_one, image_two = x, x

        # print("image_one.shape", image_one.shape)
        # print("image_two.shape", image_two.shape)

        online_proj_one, _ = self.online_encoder(img_one)       
        online_proj_two, _ = self.online_encoder(img_two)    
                                                                
        online_pred_one = self.online_predictor(online_proj_one)  
        online_pred_two = self.online_predictor(online_proj_two) 

        with paddle.no_grad(): 
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder 
            target_proj_one, _ = target_encoder(img_one)                                            
            target_proj_two, _ = target_encoder(img_two)                                            
            target_proj_one = target_proj_one.detach()                                                                 
            target_proj_two = target_proj_two.detach()             

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())    
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())    
                                                                        
        loss = loss_one + loss_two                                     
        return loss.mean()                                                                                                   



# resnet = ResNet50()
# learner = BYOL(resnet)
# for step in range(20):
#     # model = ResNet50(pretrained=True)
#     input = paddle.randn([2, 3, 256, 256],'float32')
#     # print(input.shape)

#     # transform = byol.transforms.TwoCropsTransform(T.Compose(augmentation))
#     # input = transform(input)
#     loss = learner(input)
#     print(loss)


   


 


            