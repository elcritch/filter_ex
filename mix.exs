defmodule FilterEx.MixProject do
  use Mix.Project

  def project do
    [
      app: :filter_ex,
      version: "0.2.4",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package()
    ]
  end

  defp description do
    """
    Some basic filters like Kalman filters implemented in Elixir with Nx.

    Largely based on a port of the excellent https://github.com/rlabbe/filterpy library.
    """
  end

  defp package do
    [
      files: ["lib", "mix.exs", "README*", "LICENSE*", "test"],
      maintainers: ["Jaremy Creechley"],
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/elcritch/filter_ex"}
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.9.2"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end
end
