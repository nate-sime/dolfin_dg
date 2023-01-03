

class IBP:

    def __init__(self, F, u, v, G):
        self.F = F
        self.u = u
        self.v = v
        self.G = G
        # print(f"Initialising {self}")
        # print(f"Shape F(u) = {F(u).ufl_shape}")
        # print(f"Shape G = {G.ufl_shape}")
        # print(f"Shape u = {u.ufl_shape}")
        # print(f"Shape v = {v.ufl_shape}")

