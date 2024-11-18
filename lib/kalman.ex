defmodule Filters.AdaptiveKalman do
  require Logger

  import Nx, only: [dot: 2]

  # import Matrex
  # alias MatrexNumerix.Math
  # import Matrex.Operators

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
      pP_post: self.pP |> Nx.backend_copy()
    }

    self
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

  def adaptive_eps_update_1d(self, z, opts) when is_struct(self, __MODULE__) do
    q_scale_factor = opts |> Keyword.fetch!(:q_scale_factor)
    eps_max = opts |> Keyword.fetch!(:eps_max)
    eps_alpha = opts |> Keyword.get(:eps_alpha, 0.9)

    exp_filter = %Filters.Exponential{alpha: eps_alpha, value: 0}

    {ak, ak_est, ak_res, ak_eps, _exp_filter, qvals, _count} =
      for zz <- z, reduce: {self, [], [], [], exp_filter, [], 0} do
        {ak, results, ak_res, ak_eps, exp_filter, qvals, count} ->
            # perform kalman filtering
            ak = ak |> predict() |> update(zz)

            # save data
            results = [ ak.x[0][0] |> Nx.to_number() | results ]
            ak_res = [ ak.y[0][0] |> Nx.to_number() | ak_res ]

            # y, S = cvfilter.y, cvfilter.S
            # eps = y.T @ inv(S) @ y
            # epss.append(eps)
            eps =
              Nx.transpose(ak.y)
              |> Nx.tensor(names: nil)
              |> Nx.dot(Nx.LinAlg.invert(ak.sS))
              |> Nx.dot(ak.y)
              # |> Nx.abs()
              # |> Nx.add(0.1)
              # |> Nx.log()
              |> then(& &1[0][0] |> Nx.to_number())

            exp_filter = exp_filter |> Filters.Exponential.update(eps)
            eps = exp_filter.value

            # ak_eps |> Enum.take(3)
            ak_eps = [ eps | ak_eps ]

            # next try continuous std-dev based adjustments
            {ak, count} =
              cond do
                eps > eps_max ->
                  count = count + 1
                  ak = %{ak | qQ: ak.qQ |> Nx.multiply(q_scale_factor * count) }
                  # Logger.debug("increase Q! #{inspect([count: count, eps: eps, qq: ak.qQ[0][0] |> Nx.to_number])}")
                  {ak, count}
                # eps > eps_max ->
                #   {ak, count + 1}
                count > 0 ->
                  # Logger.debug("decrease Q! #{inspect([count: count, eps: eps, qq: ak.qQ[0][0] |> Nx.to_number])}")
                  ak = %{ak | qQ: ak.qQ |> Nx.divide(q_scale_factor * count) }
                  {ak, count - 1}
                true ->
                  {ak, count}
              end

            qvals = [ ak.qQ[0][0] |> Nx.to_number | qvals ]

            {ak, results, ak_res, ak_eps, exp_filter, qvals, count}
        end

    {ak, %{estimates: ak_est |> Enum.reverse(),
           eps: ak_eps |> Enum.reverse(),
           qvals: qvals |> Enum.reverse(),
           residuals: ak_res |> Enum.reverse()}}
  end

  def adaptive_zarchan_update_1d(self, z, opts) when is_struct(self, __MODULE__) do
    q_scale_factor = opts |> Keyword.fetch!(:q_scale_factor)
    std_scale = opts |> Keyword.fetch!(:std_scale)
    phi = opts |> Keyword.get(:phi, 0.02)

    {ak, ak_est, ak_res, _phi, _count} =
      for zz <- z, reduce: {self, [], [], phi, 0} do
        {ak, results, ak_res, phi, count} ->
            # perform kalman filtering
            ak = ak |> predict() |> update(zz)

            # save data
            results = [ ak.x[0][0] |> Nx.to_number() | results ]
            ak_res = [ ak.y[0][0] |> Nx.to_number() | ak_res ]

            # y, S = cvfilter.y, cvfilter.S
            std = Nx.sqrt(ak.sS)
            # next try continuous std-dev based adjustments
            resid = abs(ak.y[0][0] |> Nx.to_number())
            scaled_std = Nx.to_number(std[0][0]) * std_scale

            # Logger.info("StdCheck: #{inspect({resid, scaled_std})}")

            {ak, phi, count} =
              cond do
                # Nx.abs(ak.y[0]) |> Nx.greater(std |> Nx.multiply(std_scale)) ->
                resid > scaled_std ->
                  phi = phi + q_scale_factor
                  ak = %{ak | qQ: ak.qQ |> Nx.add(1/q_scale_factor)}
                  # Logger.info("Increase: #{inspect(ak.qQ |> Nx.to_number)}")
                  # ak.qQ = q_discrete_white_noise(2, dt, phi)
                  {ak, phi, count + 1}
                count > 0 ->
                  phi = phi - q_scale_factor
                  ak = %{ak | qQ: ak.qQ |> Nx.subtract(1/q_scale_factor)}
                  # ak.qQ = q_discrete_white_noise(2, dt, phi)
                  # Logger.info("Decrease: #{inspect(ak.qQ)}")
                  {ak, phi, count - 1}
                true ->
                  # Logger.info("Stable: #{inspect(ak.qQ)}")
                  {ak, phi, count}
              end

            {ak, results, ak_res, phi, count}
        end

    {ak, %{estimates: ak_est |> Enum.reverse(), residuals: ak_res |> Enum.reverse()}}
  end

  def tt(x) do
    Nx.transpose(x)
  end

  def zeros({m,n}) do
    Nx.broadcast(0.0, {m,n})
  end

  @doc ~s"""
    ensure z is a (dim_z, 1) shaped vector

    ## Examples

      iex> reshape_z(3.0, 1, 1)
      Nx.tensor([ 3.0 ], names: [:x])

      iex> reshape_z(3.0, 1, 2)
      Nx.tensor([[ 3.0 ]], names: [:x, :y])

      iex> reshape_z([3.0], 1, 1)
      Nx.tensor([ 3.0 ], names: [:x])

      iex> reshape_z([3.0], 1, 2)
      Nx.tensor([[3.0]], names: [:x, :y])

      iex> reshape_z([3.0,2.0], 2, 1)
      Nx.tensor([3.0, 2.0], names: [:x])

      iex> reshape_z([3.0,2.0], 2, 2)
      Nx.tensor([[3.0], [2.0]], names: [:x, :y])

  """
  def reshape_z(z, dim_z, ndim) do
    # IO.inspect(z, label: "reshape z::")
    # IO.inspect({dim_z, ndim}, label: "reshape {dim_z, ndim}::")

    z = if is_number(z) do Nx.broadcast(z, {1,1}) else z |> Nx.tensor() end

    z = case z |> Nx.shape() do
      {n} -> z |> Nx.reshape({1,n})
      _other -> z
    end

    z = z |> Nx.tensor(names: [:y, :x])
    # IO.inspect(z, label: Zz)

    z = if z |> Nx.shape() |> elem(1) == dim_z do z |> tt() else z end
    # IO.inspect(z, label: Zzz)

    if Nx.shape(z) != {dim_z, 1} do
        raise %ArgumentError{ message: "z (shape #{Nx.shape(z)}) must be convertible to shape (#{dim_z}, 1)" }
    end

    z = if ndim == 1 do z[y: 0] else z end
    # IO.inspect(z, label: Zzza)

    z = if ndim == 0 do z[0][0] else z end
    # IO.inspect(z, label: Zzzb)

    z
  end

  @doc """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.

    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    Parameters
    -----------

    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    var : float, default=1.0
        variance in the noise

    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.

    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)

        [x x' x'' y y' y'']

        whereas `False` interleaves the dimensions

        [x y z x' y' z' x'' y'' z'']


    Examples
    --------
    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_discrete_white_noise(2, dt=0.1, var=1., block_size=3)
    array([[0.000025, 0.0005  , 0.      , 0.      , 0.      , 0.      ],
           [0.0005  , 0.01    , 0.      , 0.      , 0.      , 0.      ],
           [0.      , 0.      , 0.000025, 0.0005  , 0.      , 0.      ],
           [0.      , 0.      , 0.0005  , 0.01    , 0.      , 0.      ],
           [0.      , 0.      , 0.      , 0.      , 0.000025, 0.0005  ],
           [0.      , 0.      , 0.      , 0.      , 0.0005  , 0.01    ]])

    References
    ----------

    Bar-Shalom. "Estimation with Applications To Tracking and Navigation".
    John Wiley & Sons, 2001. Page 274.
    """
  defp q_discrete_white_noise(dim, dt=1.0, var=1.0, block_size=1, order_by_dim=true) do

    if !(dim in [2, 3, 4]) do
      raise %ArgumentError{message: "dim must be between 2 and 4"}
    end

    qQ =
      case dim do
        2 ->
          [[0.25*dt**4, 0.5*dt**3],
           [0.5*dt**3,      dt**2]]
        3 ->
          [[0.25*dt**4, 0.5*dt**3, 0.5*dt**2],
           [0.5*dt**3,      dt**2,     dt],
           [0.5*dt**2,       dt,        1]]
        _other ->
          [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
           [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
           [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
           [(dt**3)/6,  (dt**2)/2 ,  dt,        1.0]]
    end

    if order_by_dim do
      # block_diag(*[qQ]*block_size) * var
      # wtf python...
    else
      order_by_derivative(qQ, dim, block_size) * var
    end
  end

  @doc """
    Given a matrix Q, ordered assuming state space
        [x y z x' y' z' x'' y'' z''...]

    return a reordered matrix assuming an ordering of
       [ x x' x'' y y' y'' z z' y'']

    This works for any covariance matrix or state transition function

    Parameters
    ----------
    Q : np.array, square
        The matrix to reorder

    dim : int >= 1

       number of independent state variables. 3 for x, y, z

    block_size : int >= 0
        Size of derivatives. Second derivative would be a block size of 3
        (x, x', x'')


  """
  defp order_by_derivative(qQ, dim, block_size) do
    nN = dim * block_size
    dD = zeros({nN, nN})
    # qQ = array(qQ)

    for {x, i} <- Enum.with_index(qQ |> Nx.ravel()), reduce: dD do
      dD ->
        f = Nx.eye(block_size) * x
        ix = div(i, dim) * block_size
        iy = rem(i, dim) * block_size
        # dD[ix .. (ix + block_size)][iy .. (iy + block_size)] = f
        dD |> Nx.put_slice({ix, iy}, f)
    end
  end

end
