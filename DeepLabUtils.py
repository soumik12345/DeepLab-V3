from tensorflow.keras.layers import (
    DepthwiseConv2D,
    BatchNormalization,
    Lambda,
    Conv2D,
    ZeroPadding2D,
    Activation
)


class Utils:

    def sepconv_batchnorm(x, filters, prefix, stride = 1, kernel_size = 3, rate = 1, depth_activation = False, epsilon = 1e-3):
        if stride == 1:
            depth_padding = 'same'
        else:
            effective_kernel_size = kernel_size + (kernel_size - 1) * (rate - 1)
            pad = effective_kernel_size - 1
            pad_begin = pad // 2
            pad_end = pad - pad_end
            x = ZeroPadding2D((pad_begin, pad_end))(x)
            depth_padding = 'valid'
        
        if not depth_activation:
            x = Activation('relu')
        

        x = DepthwiseConv2D(
            (
                kernel_size,
                kernel_size
            ),
            strides = (
                stride,
                stride
            ),
            dilation_rate = (
                rate,
                rate
            ),
            padding = depth_padding,
            use_bias = False,
            name = prefix + '_depthwise'
        )

        x = BatchNormalization(
            name = prefix + '_depthwise_batch_norm',
            epsilon = epsilon
        )(x)

        if depth_activation:
            x = Activation('relu')(x)
        
        x = Conv2D(
            filters,
            (1, 1),
            padding = 'same',
            use_bias = False,
            name = prefix + '_pointwise'
        )

        x = BatchNormalization(
            name = prefix + '_pointwise_batch_norm',
            epsilon = epsilon
        )(x)

        if depth_activation:
            x = Activation('relu')
        
        return x