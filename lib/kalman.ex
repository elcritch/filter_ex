defmodule FilterEx.Kalman do
  require Logger

  import Nx, only: [dot: 2]
  import FilterEx.Utils
  alias FilterEx.ExpAverage

  # Kalman Paramters
  defstruct [
    :dim_x,
    :dim_z,
    :dim_u,

    :x, # state
    :pP, # uncertainty covariance
    :qQ, # process uncertainty
    :bB, # control transition matrix
    :fF, # state transition matrix
    :hH, # measurement function
    :rR, # measurement uncertainty
    :alpha, # fading memory control
    :mM, # process-measurement cross correlation
    :z,

    :kK, # kalman gain
    :y,
    :sS, # system uncertainty
    :sSI, # inverse system uncertainty

    :_I, # identity

    :x_prior,
    :pP_prior,
    :x_post,
    :pP_post,

    :kind,
    :adaptive,
  ]

  @doc """
    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u

    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

    dim_u : int (optional)
        size of the control input, if it is being used.
        Default value of 0 indicates it is not used.
  """

  def new(opts \\ []) do
    dim_x = opts |> Keyword.fetch!(:dim_x)
    dim_z = opts |> Keyword.fetch!(:dim_z)
    dim_u = opts |> Keyword.get(:dim_u, 0.0)

    if dim_x < 1.0, do: raise %ArgumentError{message: "dim_x must be 1 or greater"}
    if dim_z < 1, do: raise %ArgumentError{message: "dim_z must be 1 or greater"}
    if dim_u < 0, do: raise %ArgumentError{message: "dim_u must be 0 or greater"}

    self = %__MODULE__{
      x: zeros({dim_x, 1}),
      pP: Nx.eye(dim_x, type: :f32),               # uncertainty covariance
      qQ: Nx.eye(dim_x, type: :f32),               # process uncertainty
      bB: nil,                             # control transition matrix
      fF: Nx.eye(dim_x, type: :f32),               # state transition matrix
      hH: zeros({dim_z, dim_x}),    # measurement function
      rR: Nx.eye(dim_z, type: :f32),               # measurement uncertainty
      alpha: 1.0,                      # fading memory control
      mM: zeros({dim_x, dim_z}),    # process-measurement cross correlation
      z: Nx.broadcast(:nan, {dim_z, 1}),

      kK: zeros({dim_x, dim_z}), # kalman gain
      y: zeros({dim_z, 1}),
      sS: zeros({dim_z, dim_z}), # system uncertainty
      sSI: zeros({dim_z, dim_z}), # inverse system uncertainty
    }

    self = %{self |
      dim_x: dim_x,
      dim_z: dim_z,
      dim_u: dim_u,
      _I: Nx.eye(dim_x),
      x_prior: self.x |> Nx.backend_copy(),
      pP_prior: self.pP |> Nx.backend_copy(),
      x_post: self.x |> Nx.backend_copy(),
      pP_post: self.pP |> Nx.backend_copy(),
      kind: :regular
    }

    self
  end

  def to_eps_adaptive(self, opts) when is_struct(self, __MODULE__) do
    q_scale_factor = opts |> Keyword.fetch!(:q_scale_factor)
    eps_max = opts |> Keyword.fetch!(:eps_max)
    eps_alpha = opts |> Keyword.get(:eps_alpha, 0.9)

    %{self |
      kind: :adaptive_eps,
      adaptive: %{
        eps_filter: %ExpAverage{alpha: eps_alpha, value: 0},
        q_scale_factor: q_scale_factor,
        eps_max: eps_max,
        count: 0
      }
    }
  end

  def to_stddev_adaptive(self, opts) when is_struct(self, __MODULE__) do
    q_scale_factor = opts |> Keyword.fetch!(:q_scale_factor)
    eps_max = opts |> Keyword.fetch!(:eps_max)
    eps_alpha = opts |> Keyword.get(:eps_alpha, 0.9)

    %{self | stddev_ad: %{
        eps_filter: %ExpAverage{alpha: eps_alpha, value: 0},
        q_scale_factor: q_scale_factor,
        eps_max: eps_max,
        count: 0
    }}
  end
  def residual(self) when is_struct(__MODULE__, self) do
    self.y
  end

  def estimate(self) when is_struct(__MODULE__, self) do
    self.x
  end

  @doc """
  Predict next state (prior) using the Kalman filter state propagation
  equations.

  Parameters
  ----------

  u : np.array, default 0
      Optional control vector.

  B : np.array(dim_x, dim_u), or None
      Optional control transition matrix; a value of None
      will cause the filter to use `self.B`.

  F : np.array(dim_x, dim_x), or None
      Optional state transition matrix; a value of None
      will cause the filter to use `self.F`.

  Q : np.array(dim_x, dim_x), scalar, or None
      Optional process noise matrix; a value of None will cause the
      filter to use `self.Q`.
  """
  def predict(self, u \\ nil, bB \\ nil, fF \\ nil, qQ \\ nil) when is_struct(self, __MODULE__) do
    bB = bB || self.bB
    fF = fF || self.fF

    qQ =
      case qQ || self.qQ do
        qQ when is_number(qQ) -> Nx.eye(self.dim_x) |> Nx.multiply(qQ)
        qQ -> qQ
      end

    # x = Fx + Bu
    x =
      if bB && u do
        dot(fF, self.x) |> Nx.add(dot(bB, u))
      else
        dot(fF, self.x)
      end

    # P = FPF' + Q
    alpha_sq = self.alpha |> :math.pow(2)
    pP = (alpha_sq |> Nx.multiply(dot(dot(fF, self.pP), fF |> tt()))) |> Nx.add(qQ)

    self = %{self |
      x: x,
      pP: pP,
      qQ: qQ,
      # save priors
      x_prior: self.x |> Nx.backend_copy(),
      pP_prior: self.pP |> Nx.backend_copy()
    }

    self
  end

  @doc """
    Add a new measurement (z) to the Kalman filter.

    If z is None, nothing is computed. However, x_post and P_post are
    updated with the prior (x_prior, P_prior), and self.z is set to None.

    Parameters
    ----------
    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.

        If you pass in a value of H, z must be a column vector the
        of the correct size.

    R : np.array, scalar, or None
        Optionally provide R to override the measurement noise for this
        one call, otherwise  self.R will be used.

    H : np.array, or None
        Optionally provide H to override the measurement function for this
        one call, otherwise self.H will be used.
  """
  def update(self, z, rR \\ nil, hH \\ nil) when is_struct(self, __MODULE__) do

    # # set to None to force recompute
    # self._log_likelihood = None
    # self._likelihood = None
    # self._mahalanobis = None

    self =
      if z do
        self
      else
        %{self |
            z: Nx.broadcast(:nan, {self.dim_z, 1}) |> tt,
            x_post: self.x |> Nx.backend_copy(),
            pP_post: self.pP |> Nx.backend_copy(),
            y: zeros({self.dim_z, 1}),
        }
      end

    rR =
      case rR do
        nil -> self.rR
        rR when is_number(rR) ->
          Nx.eye(self.dim_z, type: :f32) * rR
        rR ->
          rR
      end

    {z, hH} =
      if hH do
        {z, hH}
      else
        z = reshape_z(z, self.dim_z, self.x |> Nx.shape() |> :erlang.tuple_size())
        {z, self.hH}
      end

    # y = z - Hx
    # error (residual) between measurement and prediction
    self = %{self | y: z |> Nx.subtract(dot(hH, self.x)) }

    # common subexpression for speed
    pPHT = dot(self.pP, tt(hH))

    # S = HPH' + R
    # project system uncertainty into measurement space
    self = %{self | sS: dot(hH, pPHT) |> Nx.add(rR) }
    self = %{self | sSI: Nx.LinAlg.invert(self.sS) }

    # K = PH'inv(S)
    # map system uncertainty into kalman gain
    self = %{self | kK: dot(pPHT, self.sSI) }

    # x = x + Ky
    # predict new x with residual scaled by the kalman gain
    self = %{self | x: self.x |> Nx.add(dot(self.kK, self.y)) }

    # P = (I-KH)P(I-KH)' + KRK'
    # This is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    iI_KH = self._I |> Nx.subtract(dot(self.kK, hH))
    self = %{self |
      pP: dot(dot(iI_KH, self.pP), tt(iI_KH)) |> Nx.add(dot(dot(self.kK, rR), tt(self.kK)))
    }

      # save measurement and posterior state
    self = %{self |
      z: z |> Nx.backend_copy(),
      x_post: self.x |> Nx.backend_copy(),
      pP_post: self.pP |> Nx.backend_copy()
  }

    self
  end

  def filter(self, zz, opts \\ []) when is_list(zz) and is_struct(self, __MODULE__) do
    debug = opts |> Keyword.get(:debug, false)
    scalar = opts |> Keyword.get(:scalar, true)
    kind = opts |> Keyword.get(:kind, :normal)

    getter = if scalar do &to_scalar/1 else fn x -> x end end

    {ak, ak_est, ak_res, filter_params, qvals} =
      for z <- zz, reduce: {self, [], [], [], []} do
        {ak, ak_est, ak_res, filter_params, qvals} ->
            # perform kalman filtering
          {ak, filter_params} =
            case kind do
              :normal ->
                ak = self |> predict() |> update(z)
                {ak, filter_params}
              :adaptive_eps ->
                ak = ak |> adaptive_eps(z)
                {ak, debug && [ ak.adaptive | filter_params ]}
              :adaptive_stddev ->
                ak = ak |> adaptive_stddev(z)
                {ak, debug && [ ak.adaptive | filter_params ]}
            end

          # save data
          ak_est = [ ak.x |> getter.() | ak_est ]

          if debug do
            ak_res = [ ak.y |> getter.() | ak_res ]
            qvals = [ ak.qQ |> getter.() | qvals ]
            {ak, ak_est, ak_res, filter_params, qvals}
          else
            {ak, ak_est, [], [], []}
          end
        end

    results =
      if debug do
        %{estimates: ak_est |> Enum.reverse(),
          filter_params: filter_params |> Enum.reverse(),
          qvals: qvals |> Enum.reverse(),
          residuals: ak_res |> Enum.reverse()}
      else
        %{estimates: ak_est |> Enum.reverse()}
      end

    {ak, results}
  end

  def adaptive_eps(self, z) when is_struct(self, __MODULE__) do
    unless self.eps_ad do
      raise %ArgumentError{message: "must setup eps adaptive using `to_eps_adaptive`"}
    end

    %{q_scale_factor: q_scale_factor,
      eps_filter: eps_filter,
      count: count,
      eps_max: eps_max} = self.adaptive

    # perform kalman filtering
    ak = self |> predict() |> update(z)

    # y, S = cvfilter.y, cvfilter.S
    # eps = y.T @ inv(S) @ y
    # epss.append(eps)
    eps =
      Nx.transpose(ak.y)
      |> Nx.tensor(names: nil)
      |> Nx.dot(Nx.LinAlg.invert(ak.sS))
      |> Nx.dot(ak.y)
      |> then(&to_scalar/1)

    {eps_filter, eps} = eps_filter |> ExpAverage.update(eps)

    # calculate revised Q factor based on eps threshold
    {ak, count} =
      cond do
        eps > eps_max ->
          count = count + 1
          ak = %{ak | qQ: ak.qQ |> Nx.multiply(q_scale_factor * count) }
          # Logger.debug("increase Q! #{inspect([count: count, eps: eps, qq: ak.qQ[0][0] |> Nx.to_number])}")
          {ak, count}
        count > 0 ->
          # Logger.debug("decrease Q! #{inspect([count: count, eps: eps, qq: ak.qQ[0][0] |> Nx.to_number])}")
          ak = %{ak | qQ: ak.qQ |> Nx.divide(q_scale_factor * count) }
          {ak, count - 1}
        true ->
          {ak, count}
      end

    %{ak | adaptive: %{self.adaptive | eps_filter: eps_filter, count: count, eps: eps}}
  end

  def adaptive_stddev(self, z) when is_struct(self, __MODULE__) do
    unless self.eps_ad do
      raise %ArgumentError{message: "must setup eps adaptive using `to_eps_adaptive`"}
    end

    %{q_scale_factor: q_scale_factor,
      std_scale: std_scale,
      count: count,
      phi: phi} = self.adaptive


    # perform kalman filtering
    self = self |> predict() |> update(z)

    # y, S = cvfilter.y, cvfilter.S
    std = Nx.sqrt(self.sS)
    # next try continuous std-dev based adjustments
    resid = abs(self.y |> to_scalar)
    scaled_std = to_scalar(std) * std_scale

    # Logger.info("StdCheck: #{inspect({resid, scaled_std})}")

    {ak, phi, count} =
      cond do
        # Nx.abs(ak.y[0]) |> Nx.greater(std |> Nx.multiply(std_scale)) ->
        resid > scaled_std ->
          phi = phi + q_scale_factor
          self = %{self | qQ: self.qQ |> Nx.add(1/q_scale_factor)}
          # Logger.info("Increase: #{inspect(ak.qQ |> Nx.to_number)}")
          # ak.qQ = q_discrete_white_noise(2, dt, phi)
          {self, phi, count + 1}
        count > 0 ->
          phi = phi - q_scale_factor
          self = %{self | qQ: self.qQ |> Nx.subtract(1/q_scale_factor)}
          # ak.qQ = q_discrete_white_noise(2, dt, phi)
          # Logger.info("Decrease: #{inspect(ak.qQ)}")
          {self, phi, count - 1}
        true ->
          # Logger.info("Stable: #{inspect(ak.qQ)}")
          {self, phi, count}
      end

    %{ak | adaptive: %{self.adaptive | phi: phi, count: count}}
  end

end
