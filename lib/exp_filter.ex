defmodule FilterEx.ExpAverage do

    # Kalman Paramters
    defstruct [
      :alpha,
      :value,
    ]

    def update(self, y) when is_number(y) and is_struct(self, __MODULE__) do
      self = %{self | value: self.value + self.alpha * (y - self.value)}
      {self, self.value}
    end

    def update(self, value) when is_list(value) and is_struct(self, __MODULE__) do
      {self, values} =
        for yy <- value, reduce: {self, []} do
          {self, prev} ->
            self = self |> update(yy)
            {self, [ self.value | prev ]}
        end

      {self, values |> Enum.reverse()}
    end

end
