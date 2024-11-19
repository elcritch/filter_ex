defmodule FilterExTest do
  use ExUnit.Case
  # doctest FilterEx
  doctest FilterEx.Utils

  test "kalman test" do

    %{random_data: random_data, xdata: xdata, ydata: ydata} = generate_data()

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
    {kalman, results} = kalman |> FilterEx.Kalman.filter(random_data)

    IO.inspect(results, label: Kalman.Results)
  end

  def generate_data(state \\ %{}) do
    # ex_unit_seed = ExUnit.configuration()[:seed]
    :rand.seed(:exsss, {1, 2, 3})

    n = 120
    xdata = 1..(2*n) |> Enum.map(&(1.0*&1))
    data_init = 1..n |> Enum.map(fn _ -> 20.0 end)
    data_then = 1..n |> Enum.map(fn _ -> 22.0 end)
    data = data_init ++ data_then
    random_data_init = 1..n |> Enum.map(fn _ -> :rand.normal(20.0, 0.03) end)
    random_data_then = 1..n |> Enum.map(fn _ -> :rand.normal(22.0, 0.03) end)
    random_data = random_data_init ++ random_data_then

    random_data =
      random_data
      |> List.replace_at(20, 23.55)


    state
    |> Map.put(:random_data, random_data)
    |> Map.put(:xdata, xdata)
    |> Map.put(:ydata, data)
  end
end
