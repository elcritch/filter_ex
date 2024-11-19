defmodule FilterEx.ExpAverage do

    @moduledoc """
    Simple exponential moving average.

    ## Examples

        iex> exp_filter = %FilterEx.ExpAverage{alpha: 0.2, value: 0.0}
        ...> {exp_filt, eps} = exp_filter |> FilterEx.ExpAverage.update(1.2)
        {%FilterEx.ExpAverage{alpha: 0.2, value: 0.24}, 0.24}

        iex> %{random_data: random_data} = FilterExTest.generate_data(3)
        ...> exp_filter = %FilterEx.ExpAverage{alpha: 0.2, value: 22.0}
        ...> {exp_filt, eps} = exp_filter |> FilterEx.ExpAverage.filter(random_data)
        {%FilterEx.ExpAverage{alpha: 0.2, value: 25.450578740346586}, [21.647465048364356, 21.30137413642628, 21.063182022003538, 22.841878849653416, 24.293343765404888, 25.450578740346586]}

    """
    # Kalman Paramters
    defstruct [
      :alpha,
      :value,
    ]

    def update(self, y) when is_number(y) and is_struct(self, __MODULE__) do
      self = %{self | value: self.value + self.alpha * (y - self.value)}
      {self, self.value}
    end

    def filter(self, value) when is_list(value) and is_struct(self, __MODULE__) do
      {self, values} =
        for yy <- value, reduce: {self, []} do
          {self, prev} ->
            {self, value} = self |> update(yy)
            {self, [ value | prev ]}
        end

      {self, values |> Enum.reverse()}
    end

end
