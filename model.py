from dataclasses import dataclass

import tensorflow as tf
import tensorflow.keras as K

from einops import rearrange


class Attention(K.Model):
    """Attention.

    Implements multi-head attention (Vaswani et al., 2017).

    Example
    -------
    >>> module = Attention(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ... )
    >>> x = tnp.random.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self, 
        *,
        embedding_dimension: int, 
        number_of_heads: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding embedding dimension.
        number_of_heads : int
            The number of heads.
        """

        super().__init__()

        self.number_of_heads = number_of_heads
        
        self.dense_1 = K.layers.Dense(
            units=embedding_dimension * 3, 
            use_bias=False,
        )

        self.dense_2 = K.layers.Dense(
            units=embedding_dimension, 
            use_bias=False,
        )
    
    def call(self, x: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Call the module.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        mask : tf.Tensor
            The attention mask (e.g. a causal mask).

        Returns
        -------
        x : tf.Tensor
            The output tensor.
        """

        q, k, v = rearrange(
            tensor=self.dense_1(x), 
            pattern='b t (n h e) -> n b t h e', 
            n=3, 
            h=self.number_of_heads,
        )

        score = tf.nn.softmax(tf.einsum('bthe,bshe->bhts', q, k) + mask)
        x = tf.einsum('bhts,bthe->bthe', score, v)
        x = self.dense_2(rearrange(x, 'b t h e -> b t (h e)'))

        return x


class TransformerBlock(K.Model):
    """Transformer block.

    Implements a transformer block (Radford et al., 2019).

    Example
    -------
    >>> module = TransformerBlock(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ... )
    >>> x = tnp.random.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self, 
        *,
        embedding_dimension: int, 
        number_of_heads: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding embedding dimension.
        number_of_heads : int
            The number of heads.
        """

        super().__init__()

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )

        self.layer_normalization_1 = K.layers.LayerNormalization()
        self.layer_normalization_2 = K.layers.LayerNormalization()

        self.mlp = K.Sequential()
        self.mlp.add(K.layers.Dense(units=embedding_dimension * 4))
        self.mlp.add(K.layers.GELU())
        self.mlp.add(K.layers.Dense(units=embedding_dimension))
    
    def call(self, x: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Call the module.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        mask : tf.Tensor
            The attention mask (e.g. a causal mask).
        
        Returns
        -------
        x : tf.Tensor
            The output tensor.
        """

        x = x + self.attention(self.layer_normalization_1(x), mask=mask)
        x = x + self.mlp(self.layer_normalization_2(x))

        return x


@dataclass(frozen=True)
class TransformerConfiguration:
    embedding_dimension: int
    number_of_heads: int
    number_of_blocks: int


class Transformer(K.Model):
    """Transformer.

    Implements a transformer (Radford et al., 2019).

    Example
    -------
    >>> configuration = TransformerConfiguration(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ...     number_of_blocks=16,
    ... )
    >>> module = Transformer(configuration=configuration)
    >>> x = tnp.random.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self, *, configuration: TransformerConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : TransformerConfiguration
            The module configuration.
        """

        self.blocks = K.Sequential()

        for _ in range(configuration.number_of_blocks):
            self.blocks.add(
                TransformerBlock(
                    embedding_dimension=configuration.embedding_dimension,
                    number_of_heads=configuration.number_of_heads,
                ),
            )
    
    def call(self, x: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Call the module.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        mask : tf.Tensor
            The attention mask (e.g. a causal mask).
        
        Returns
        -------
        x : tf.Tensor
            The output tensor.
        """

        x = self.blocks(x)

        return x
