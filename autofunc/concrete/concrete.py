# +
import autofunc.abstract as ab


def concrete(cls: type, cfg: dict[type, type], suffix: str = "") -> type:
    bases: tuple[type, ...] = (cls,)
    for n in ab.__all__:
        t: type = getattr(ab, n)
        if t in {ab.Abs, ab.Tens}:
            continue
        if issubclass(cls, t):
            b = cfg[t]
            bases = (b, *bases)
    name = f"{cls.__name__}_{suffix}"
    return type(name, bases, {})
