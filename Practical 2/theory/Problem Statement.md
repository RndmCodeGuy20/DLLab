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

1. Adaptive Gradient Descent
2. RMSProp

To approximate certain values of x and y, so that the value of z is minimum.

### Equation of Adaptive Gradient Descent

$$
\begin{aligned}
w_{t+1} &= w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla_w L(w_t) \\
b_{t+1} &= b_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla_b L(b_t)
\end{aligned}
$$

### Equation of RMSProp

$$
\begin{aligned}
G_t &= \gamma G_{t-1} + (1 - \gamma) \nabla_w L(w_t)^2 \\
w_{t+1} &= w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla_w L(w_t)
\end{aligned}
