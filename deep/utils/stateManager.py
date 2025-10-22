from typing import Any

class _Obj:
    """
    This class keeps track of every modification of each subState.
    """
    def __init__(self, v: Any, allow_update: bool = True, allow_dtyping: bool = True):
        """

        :param v: Value of the subState
        :param allow_update: Whether to allow the update of the subState.  If false, it will be immutable.
        :param allow_dtyping: Allow dynamic data typing.  If False, the subState cannot change of datatype during updates.
        """
        self.t = [type(v)]
        self.v = v
        self._allow_update = allow_update
        self._allow_dtyping = allow_dtyping
        self._updates = 1
        self._reads = 0

    def update(self, new_value):
        """
        Used to update the internal value of the subState
        :param new_value: The new value to set
        :return: None
        """
        if isinstance(new_value, self.dtype) or self._allow_dtyping:
            if not isinstance(new_value, self.dtype):
                self.t.append(type(new_value))
            if self._allow_update:
                self.v = new_value
                self._updates += 1
            else:
                raise RuntimeError("Cannot update state because allow_update is turned off.")
        else:
            raise RuntimeError("Cannot change data type of state if dynamic typing is not enabled.")

    def get(self):
        """
        Return the internal value
        :return:
        """
        self._reads += 1
        return self.v

    @property
    def dtype(self) -> type:
        """
        Return the datatype of the actual value.
        :return: DType
        """
        return self.t[-1]

    def summary(self) -> dict:
        """
        Return a dictionary of the internal data.
        :return: internal data as dict.
        """
        return {
            "Types": self.t,
            "updates": self._updates,
            "reads": self._reads,
            "data": self.v
        }
    def __str__(self):
        internal = self.summary()
        return f"({internal['Types'][-1]}, u[{internal['updates']}], r[{internal['reads']}]) value: {internal['data']}"
    def __repr__(self):
        internal = self.summary()
        return f"({internal['Types'][-1]}, u[{internal['updates']}], r[{internal['reads']}])"


class StateManager:
    """
    This object is a state manager hence its name.  It is designed to contain state of different parts of application
    and these states can be used from anywhere.  Its UX is 'pythonic', meaning it is easy:)

    How it works:
        1. Importe the State.
        2. Set options if default ones aren't ok
        3. Add attributes to the object like you would on a normal object.
        4. Retrieve these attributes from anywhere.

    Example:
        >>> from utils import State

        >>> State.set_options(allow_update=True, allow_dtyping=True)

        >>> State.epoch_loss = 0.99

        >>> # In other file that imported State:

        >>> loss = State.epoch_loss

    There are a lot of useful other functionalities to find out!
    """
    _internal = {}
    _params = {"allow_update": True, "allow_dtyping": True}

    def __setattr__(self, key, value):
        internal = object.__getattribute__(self, "_internal")
        obj: _Obj = internal.get(key)
        if obj is None:
            internal[key] = _Obj(value, **object.__getattribute__(self, "_params"))
        else:
            internal[key].update(value)

    def __getattr__(self, key: str):
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        obj: _Obj = object.__getattribute__(self, "_internal").get(key)
        if obj is None:
            raise AttributeError(f"'StateManager' object has no attribute '{key}'")
        else:
            return obj.get()

    def set_options(self, **kwargs):
        """
        Set options that are passed to the _Obj object.  These options are applied to every NEW subState.  SubStates
        created before setting those options wille keep their options.  See _Obj documentation for more details.
        :param kwargs: Options to set.
            Supported are:
                allow_update: bool
                allow_dtyping: bool
        :return: None
        """
        params = object.__getattribute__(self, "_params")
        for key, value in kwargs.items():
            if params.get(key) is not None:
                params[key] = value
            else:
                raise RuntimeError(f"Invalid option key: {key}")

    def get_type(self, key) -> type:
        """
        Get the datatype of a subState.
        :param key: The name of the subState
        :return: The dataType
        """
        obj: _Obj = object.__getattribute__(self, "_internal").get(key)
        if obj is None:
            raise AttributeError(f"'StateManager' object has no attribute '{key}'")
        else:
            return obj.dtype

    def summary(self) -> dict:
        """
        This method will return the internal state of the StateManager object.
        :return: The internal dictionary containing every subStates.
        """
        return object.__getattribute__(self, "_internal")

    def __str__(self):
        summary = self.summary()
        keys = list(summary.keys())
        max_key_len = len(max(keys, key=lambda x: len(x)))
        s = "StateManagerObject:\n"
        for key in keys:
            s += f'\t{key}: {" " * (max_key_len - len(key))}{repr(summary[key])}\n'
        s = s.rstrip("\n")
        return s

    def _extract_warns(self) -> dict:
        # Check if some values were written more than they were read.  (This means that a value is ignored somewhere,
        # which might lead to a bug, or confusion during code maintenance/review)
        summary = self.summary()
        warns = {}
        for key, value in summary.items():
            s = value.summary()
            update, reads = s["updates"], s["reads"]
            if reads < update:
                warns[key] = (update, reads)
        return warns

    def has_warning(self) -> bool:
        """
        This function will verify if some variable are updated more than they are read.  If this happens, it would mean
        that some operation's results are never used.  This could lead to hard to detect bugs or confusion during code
        maintenance/review
        Returns: True if there are such variable, False otherwise

        """
        warns = self._extract_warns()
        return len(warns) > 0

    def warning(self) -> str:
        warns = self._extract_warns()
        if len(warns) == 0:
            return ""

        if len(warns) == 1:
            s = f"Warning, you have {len(warns)} variable that is updated more than it is read.  This could lead to " \
                f"undesirable comportment:\n"
        else:    # More than 1
            s = f"Warning, you have {len(warns)} variable that are updated more than they are read.  This could " \
                f"lead to undesirable comportment:\n"
        for var_name, (u, r) in warns.items():
            s += f"   -> [{var_name}] is updated {u} times, but is read only {r} times\n"
        return s



# This is the state of the application.
State = StateManager()

if __name__ == "__main__":
    state = StateManager()
    state.set_options(allow_update=True, allow_dtyping=True)
    state.hello = "world"
    state.test = 7.
    state.tmp = 42
    state.anticonstitutionellement = "long onneeee"
    print(state)