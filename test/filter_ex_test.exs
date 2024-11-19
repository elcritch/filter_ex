defmodule FilterExTest do
  alias FilterEx.Kalman
  use ExUnit.Case
  # doctest FilterEx
  doctest FilterEx.Utils
  doctest FilterEx.Kalman

  test "kalman test" do

    %{n: n, random_data: random_data} = generate_data()

    kalman =
      FilterEx.Kalman.new(dim_x: 1, dim_z: 1, dim_u: 1)
      |> then(& %{&1 |
        x: Nx.tensor([[20.0]]),       # initial state (location and velocity)
        fF: Nx.tensor([[1.0]]),    # state transition matrix
        hH: Nx.tensor([[1.0]]),    # Measurement function
        pP: &1.pP |> Nx.multiply(1.0),  # covariance matrix
        rR: 1.0,                       # state uncertainty
        qQ: 1.0/381.0                  # process uncertainty
      })

    # IO.inspect(kalman, label: Kalman)
    {_kalman, %{estimates: estimates}} = kalman |> FilterEx.Kalman.filter(random_data)

    n1 = estimates |> Enum.take(n) |> Nx.tensor()
    n2 = estimates |> Enum.take(-div(n, 2)) |> Nx.tensor()
    n2a = estimates |> Enum.drop(n) |> Enum.take(div(n,4)) |> Nx.tensor()

    # IO.inspect(kalman, label: Kalman.Kalman)
    # IO.inspect(n1, label: Kalman.Results.N1, limit: :infinity)
    # IO.inspect(n2, label: Kalman.Results.N2, limit: :infinity)
    n1 = Nx.sum(n1) |> FilterEx.Utils.to_scalar() |> Kernel./(n)
    n2 = Nx.sum(n2) |> FilterEx.Utils.to_scalar() |> Kernel./(n/2)
    n2a = Nx.sum(n2a) |> FilterEx.Utils.to_scalar() |> Kernel./(n/4)

    # IO.inspect(n1, label: Kalman.Results.N1, limit: :infinity)
    # IO.inspect(n2, label: Kalman.Results.N2, limit: :infinity)
    assert_in_delta n1, 20.0, 0.9
    assert_in_delta n2, 30.0, 0.9
    assert_in_delta n2a, 24.0, 0.9
  end

  test "adaptive kalman test" do

    %{n: n, random_data: random_data} = generate_data()

    kalman =
      FilterEx.Kalman.new(dim_x: 1, dim_z: 1, dim_u: 1)
      |> FilterEx.Kalman.set(
        x: 20.0,       # initial state (location and velocity)
        fF: 1.0,    # state transition matrix
        hH: 1.0,    # Measurement function
        rR: 1.0,                       # state uncertainty
        qQ: 1.0/381.0                  # process uncertainty
      )
      |> Kalman.to_eps_adaptive(q_scale_factor: 3.1, eps_max: 1.0)

    # IO.inspect(kalman, label: Kalman)
    {_kalman, %{estimates: estimates}} = kalman |> FilterEx.Kalman.filter(random_data)

    n1 = estimates |> Enum.take(n) |> Nx.tensor()
    n2 = estimates |> Enum.take(-div(n, 2)) |> Nx.tensor()
    n2a = estimates |> Enum.drop(n) |> Enum.take(div(n,4)) |> Nx.tensor()

    # IO.inspect(kalman, label: Kalman.Kalman)
    # IO.inspect(n1, label: Kalman.Results.N1, limit: :infinity)
    # IO.inspect(n2, label: Kalman.Results.N2, limit: :infinity)
    n1 = Nx.sum(n1) |> FilterEx.Utils.to_scalar() |> Kernel./(n)
    n2 = Nx.sum(n2) |> FilterEx.Utils.to_scalar() |> Kernel./(n/2)
    n2a = Nx.sum(n2a) |> FilterEx.Utils.to_scalar() |> Kernel./(n/4)

    # IO.inspect(n1, label: Kalman.Results.N1, limit: :infinity)
    # IO.inspect(n2, label: Kalman.Results.N2, limit: :infinity)
    # IO.inspect(n2a, label: Kalman.Results.N2A, limit: :infinity)
    assert_in_delta n1, 20.0, 0.9
    assert_in_delta n2, 30.0, 0.9
    # we get a value much closer to the second average very quickly!
    assert_in_delta n2a, 29.0, 0.9
  end

  def generate_data(n \\ 80) do
    state = %{}
    # ex_unit_seed = ExUnit.configuration()[:seed]
    :rand.seed(:exsss, {1, 2, 3})
    data_init = 1..n |> Enum.map(fn _ -> 20.0 end)
    data_then = 1..n |> Enum.map(fn _ -> 22.0 end)
    data = data_init ++ data_then
    random_data_init = 1..n |> Enum.map(fn _ -> :rand.normal(20.0, 0.03) end)
    random_data_then = 1..n |> Enum.map(fn _ -> :rand.normal(30.0, 0.03) end)
    random_data = random_data_init ++ random_data_then

    # random_data =
    #   random_data
    #   |> List.replace_at(20, 23.55)

    state
    |> Map.put(:n, n)
    |> Map.put(:random_data, random_data)
    |> Map.put(:raw_data, data)
  end
end
