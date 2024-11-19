# FilterEx

Some basic filters implemented in Elixir with Nx. Currently only Kalman, Adaptive Kalman, and exponential moving filters are implemented.

The Kalman filter is based on a port of the excellent [filterpy](https://github.com/rlabbe/filterpy) library. It's only a partial port. Furthermore it's really only been tested on 1D, but should in theory work on higher dim Kalman's as well. PR's welcome!

Checkout @rlabbe's excellent book [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python). The derivation of Kalman filters from Gaussians is a handy. 

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `filter_ex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:filter_ex, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/filter_ex>.

