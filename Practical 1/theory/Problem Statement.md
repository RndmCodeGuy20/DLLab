<style>
h1, h2, h3 {
font-family: "Inria Serif Light", sans-serif;
}
body { 
font-family: "IBM Plex Sans", sans-serif;
font-weight: 400;
}

code { 
font-weight: 600;
}
</style>

Write a program to implement:

1. Vanilla Gradient Descent
2. Momentum Based Gradient Descent

To approximate certain values of x and y.

```python
x = [1, 3.5, 6]
y = [4, 5.5, 9]

lr = 0.05
```

## Vanilla Gradient Descent

Given a set of inputs $X$ and $Y$, we need to fit a linear function $y = wx + b$ so that the error is minimum.


