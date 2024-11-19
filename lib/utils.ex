defmodule FilterEx.Utils do

  def tt(x) do
    Nx.transpose(x)
  end

  def zeros({m,n}, def \\ 0.0) do
    Nx.broadcast(def, {m,n})
  end

  @doc """
  Convert scalar, list, 1d-tensor to a 2d tensor.

  iex> 1.0 |> FilterEx.Utils.to_tensor_2d()
  Nx.tensor([[1.0]])

  iex> [1.0, 2.0] |> FilterEx.Utils.to_tensor_2d()
  Nx.tensor([[1.0], [2.0]])

  iex> [[1.0, 2.0], [3.0, 4.0]] |> FilterEx.Utils.to_tensor_2d()
  Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

  """
  def to_tensor_2d(value) do
    case value do
      value when is_number(value) ->
        Nx.tensor([[value]], type: :f32)
      [] ->
        raise %ArgumentError{message: "unable to convert tensor shape to 2d"}
      [v | _] when is_list(value) and (is_number(v) or v == nil or v == :nan or v == :inf or v == :neg_inf) ->
        Nx.tensor([value], type: :f32)
        |> Nx.transpose()
      value when is_list(value) ->
        Nx.tensor(value, type: :f32)
      %Nx.Tensor{shape: {_}} ->
        Nx.tensor([value], type: :f32)
      %Nx.Tensor{shape: {_, _}} ->
        value
      %Nx.Tensor{} ->
        raise %ArgumentError{message: "unable to convert tensor shape to 2d"}
    end
  end

  @doc ~s"""
    ensure z is a (dim_z, 1) shaped vector

    ## Examples

        iex> FilterEx.Utils.reshape_z(3.0, 1, 1)
        Nx.tensor([ 3.0 ], names: [:x])

        iex> FilterEx.Utils.reshape_z(3.0, 1, 2)
        Nx.tensor([[ 3.0 ]], names: [:x, :y])

        iex> FilterEx.Utils.reshape_z([3.0], 1, 1)
        Nx.tensor([ 3.0 ], names: [:x])

        iex> FilterEx.Utils.reshape_z([3.0], 1, 2)
        Nx.tensor([[3.0]], names: [:x, :y])

        iex> FilterEx.Utils.reshape_z([3.0,2.0], 2, 1)
        Nx.tensor([3.0, 2.0], names: [:x])

        iex> FilterEx.Utils.reshape_z([3.0,2.0], 2, 2)
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
        iex> # constant velocity model in a 3D world with a 10 Hz update rate
        ...> FilterEx.Utils.q_discrete_white_noise(2, dt: 0.1, var: 1.0, block_size: 3)
        Nx.tensor([[0.000025, 0.0005  , 0.0      , 0.0      , 0.0      , 0.0      ],
                  [0.0005  , 0.01    , 0.0      , 0.0      , 0.0      , 0.0      ],
                  [0.0      , 0.0      , 0.000025, 0.0005  , 0.0      , 0.0      ],
                  [0.0      , 0.0      , 0.0005  , 0.01    , 0.0      , 0.0      ],
                  [0.0      , 0.0      , 0.0      , 0.0      , 0.000025, 0.0005  ],
                  [0.0      , 0.0      , 0.0      , 0.0      , 0.0005  , 0.01    ]
        ], type: :f32)

        iex> FilterEx.Utils.q_discrete_white_noise(2, dt: 0.1, var: 1.0, block_size: 3, order_by_dim: false)
        Nx.tensor([
            [2.499999936844688e-5, 0.0, 0.0, 5.000000237487257e-4, 0.0, 0.0],
            [0.0, 2.499999936844688e-5, 0.0, 0.0, 5.000000237487257e-4, 0.0],
            [0.0, 0.0, 2.499999936844688e-5, 0.0, 0.0, 5.000000237487257e-4],
            [5.000000237487257e-4, 0.0, 0.0, 0.009999999776482582, 0.0, 0.0],
            [0.0, 5.000000237487257e-4, 0.0, 0.0, 0.009999999776482582, 0.0],
            [0.0, 0.0, 5.000000237487257e-4, 0.0, 0.0, 0.009999999776482582]
        ], type: :f32)

    References
    ----------

    Bar-Shalom. "Estimation with Applications To Tracking and Navigation".
    John Wiley & Sons, 2001. Page 274.
    """
  def q_discrete_white_noise(dim, opts) do

    dt = opts |> Keyword.get(:dt, 1.0)
    var = opts |> Keyword.get(:var, 1.0)
    block_size = opts |> Keyword.get(:block_size, 1)
    order_by_dim = opts |> Keyword.get(:order_by_dim, true)
    mtyp = opts |> Keyword.get(:mtyp, :f32)

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
    |> Nx.tensor(type: mtyp)

    # IO.inspect(qQ, label: "q_diag: qq: ")
    if order_by_dim do
      # block_diag(*[qQ]*block_size) * var # wtf python...
      mats = 1..block_size |> Enum.map(fn _ -> qQ end)
      block_diag(mats) |> Nx.multiply(var)
    else
      order_by_derivative(qQ, dim, block_size) |> Nx.multiply(var)
    end
  end

  @doc """
  Creates a block diagonal matrix from a list of matrices using Nx.

  ## Examples

      iex> qQ = Nx.tensor([[1, 2], [3, 4]], type: :f32)
      ...> matrices = [qQ, qQ, qQ]  # List of matrices to place on the diagonal
      ...> FilterEx.Utils.block_diag(matrices)
      Nx.tensor([
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 3.0, 4.0]
      ], type: :f32)

  """
  def block_diag(matrices) when is_list(matrices) do
    # Calculate the total size of the resulting matrix
    # IO.inspect(matrices, label: "block_diag:matrices")
    {total_rows, total_cols, {styp, ssz}} =
      for mat <- matrices, reduce: {0, 0, nil} do
        {rows, cols, _typ} ->
          rows = rows + elem(Nx.shape(mat), 0)
          cols = cols + elem(Nx.shape(mat), 1)
          {rows, cols, Nx.type(mat)}
      end

    # IO.inspect({total_rows, total_cols, {styp, ssz}}, label: "block_diag:total indx")
    # Initialize an empty matrix of zeros
    result =
      Nx.tensor(0, type: :"#{styp}#{ssz}")
      |> Nx.broadcast({total_rows, total_cols})

    # Fill in each block along the diagonal
    for mat <- matrices, reduce: {result, 0, 0} do
      {acc, start_row, start_col} ->
        {rows, cols} = Nx.shape(mat)
        updated_matrix =
          acc
          |> Nx.put_slice([start_row, start_col], mat)

      {updated_matrix, start_row + rows, start_col + cols}
    end
    |> elem(0)
  end

  @doc """
  Flattens the input tensor into a 1D tensor using Nx.

  ## Examples

      iex> Nx.tensor([[1, 2, 3], [4, 5, 6]], type: :f32) |> FilterEx.Utils.ravel()
      Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], type: :f32)

  """
  def ravel(tensor) do
    # Get the total number of elements in the tensor
    num_elements = Nx.size(tensor)

    # Reshape the tensor into a 1D tensor
    Nx.reshape(tensor, {num_elements})
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
  def order_by_derivative(qQ, dim, block_size) do
    nN = dim * block_size
    dD = zeros({nN, nN})
    # qQ = array(qQ)

    for {x, i} <- Enum.with_index(qQ |> ravel() |> Nx.to_list()), reduce: dD do
      dD ->
        f = Nx.eye(block_size) |> Nx.multiply(x)
        ix = div(i, dim) * block_size
        iy = rem(i, dim) * block_size
        # dD[ix .. (ix + block_size)][iy .. (iy + block_size)] = f
        dD |> Nx.put_slice([ix, iy], f)
    end
  end

  @doc """
  Get scalar number from a 1, 2, or 3 dimension tensor from zeroth index.

  """
  def to_scalar(mat) do
    case mat |> Nx.shape() do
      {} ->
        mat
      {_n} ->
        mat[0]
      {_m, _n} ->
        mat[0][0]
      {_m, _n, _o} ->
        mat[0][0][0]
    end
    |> Nx.to_number()
  end

end
