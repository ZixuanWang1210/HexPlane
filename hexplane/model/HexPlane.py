import torch
from torch.nn import functional as F

from hexplane.model.HexPlane_Base import HexPlane_Base


class HexPlane(HexPlane_Base):
    """
    A general version of HexPlane, which supports different fusion methods and feature regressor methods.
    """

    def __init__(self, aabb, gridSize, device, time_grid, near_far, **kargs):
        super().__init__(aabb, gridSize, device, time_grid, near_far, **kargs)

    def init_planes(self, res, device):
        """
        Initialize the planes. density_plane is the spatial plane while density_line_time is the spatial-temporal plane.
        """
        # self.density_plane中的每个元素都是一个四维张量，用于表示一个特定的空间平面。
        # 根据组件的数量和体素网格大小确定
        self.density_plane, self.density_line_time = self.init_one_hexplane(
            self.density_n_comp, self.gridSize, device
        )
        self.app_plane, self.app_line_time = self.init_one_hexplane(
            self.app_n_comp, self.gridSize, device
        )

        if (
            self.fusion_two != "concat"
        ):  # if fusion_two is not concat, then we need dimensions from each paired planes are the same.
            assert self.app_n_comp[0] == self.app_n_comp[1]
            assert self.app_n_comp[0] == self.app_n_comp[2]

        # We use density_basis_mat and app_basis_mat to project extracted features from HexPlane to density_dim/app_dim.
        # density_basis_mat and app_basis_mat are linear layers, whose input dims are calculated based on the fusion methods.
        
        # 使用density_basis_mat和app_basis_mat将从HexPlane提取的特征投影到density_dim/app_dim。
        # density_basis_mat和app_basis_mat是线性层，其输入维度基于融合方法进行计算。
        if self.fusion_two == "concat":
            if self.fusion_one == "concat":
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp) * 2, self.density_dim, bias=False
                ).to(device)
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp) * 2, self.app_dim, bias=False
                ).to(device)
            else:
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp), self.density_dim, bias=False
                ).to(device)
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp), self.app_dim, bias=False
                ).to(device)
        else:
            self.density_basis_mat = torch.nn.Linear(
                self.density_n_comp[0], self.density_dim, bias=False
            ).to(device)
            self.app_basis_mat = torch.nn.Linear(
                self.app_n_comp[0], self.app_dim, bias=False
            ).to(device)

        # Initialize the basis matrices
        # 初始化为 1 / density_dim，固定值
        with torch.no_grad():
            weights = torch.ones_like(self.density_basis_mat.weight) / float(
                self.density_dim
            )
            self.density_basis_mat.weight.copy_(weights)

    # 平面的初始化
    # 【n_component是一个初始化为8的整数或数组】每一个密度相关的空间或时空平面都有8个分量（8个组件）
    def init_one_hexplane(self, n_component, gridSize, device):
        plane_coef, line_time_coef = [], []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            # 平面/时间系数 随机初始化：正态分布+扰动
            # plane_coef 结构是（1, n_component[i], gridSize[mat_id_1] 空间网格索引, gridSize[mat_id_0]）4维
            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
            # line_time_coef结构是（1, n_component[i], gridSize[vec_id] 时间网格索引, self.time_grid 时间网格索引）
            line_time_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn((1, n_component[i], gridSize[vec_id], self.time_grid))
                    + self.init_shift
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_time_coef
        ).to(device)

    def get_optparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.density_line_time,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.app_line_time,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
            {
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn,
                "lr_org": cfg.lr_app_nn,
            },
        ]

        if isinstance(self.app_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]

        return grad_vars

    def compute_densityfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Compuate the density features of sampled points from density HexPlane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            density: (N) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat = [], []
        # Extract features from six feature planes.
        for idx_plane in range(len(self.density_plane)):
            # 使用PyTorch的grid_sample函数，在给定的坐标（plane_coord[[idx_plane]]）上对密度平面进行网格采样。
            # align_corners参数用于指定网格采样中坐标的对齐方式。
            # 最后，通过view方法改变张量形状以匹配xyz_sampled的形状。
            # 这一步是空间平面特征的提取。
            plane_feat.append(
                F.grid_sample(
                    self.density_plane[idx_plane],  # 要进行网格采样的密度平面
                    plane_coord[[idx_plane]],  # 对应于该密度平面的坐标
                    align_corners=self.align_corners,  # 坐标对齐参数
                ).view(-1, *xyz_sampled.shape[:1])  # 改变张量形状
            )
            # Spatial-Temoral Feature: Grid sampling on density line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.density_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            #
        # 直接堆叠特征
        plane_feat, line_time_feat = torch.stack(plane_feat, dim=0), torch.stack(
            line_time_feat, dim=0
        )

        # Fusion One 第一次融合 融合时空特征
        if self.fusion_one == "multiply":
            inter = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            inter = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            inter = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            raise NotImplementedError("no such fusion type")

        # Fusion Two 第二次融合
        if self.fusion_two == "multiply":
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")


        #特征降维
        inter = self.density_basis_mat(inter.T)  # Feature Projection

        return inter

    # 和上面的类似，这个是计算appearance特征的
    def compute_appfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Compuate the app features of sampled points from appearance HexPlane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            app_feature: (N, self.app_dim) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat = [], []
        for idx_plane in range(len(self.app_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.app_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        plane_feat, line_time_feat = torch.stack(plane_feat), torch.stack(
            line_time_feat
        )

        # Fusion One
        if self.fusion_one == "multiply":
            inter = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            inter = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            inter = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        if self.fusion_two == "multiply":
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.app_basis_mat(inter.T)  # Feature Projection

        return inter

    def TV_loss_density(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.density_plane)):
            total = (
                total + reg(self.density_plane[idx]) + reg2(self.density_line_time[idx])
            )
        return total

    # TV损失函数鼓励模型生成平滑的结果
    def TV_loss_app(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) + reg2(self.app_line_time[idx])
        return total

    def L1_loss_density(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line_time[idx]))
            )
        return total

    def L1_loss_app(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.app_plane[idx]))
                + torch.mean(torch.abs(self.app_line_time[idx]))
            )
        return total

    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, line_time_coef, res_target, time_grid):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
            line_time_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_time_coef[i].data,
                    size=(res_target[vec_id], time_grid),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )

        return plane_coef, line_time_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        self.app_plane, self.app_line_time = self.up_sampling_planes(
            self.app_plane, self.app_line_time, res_target, time_grid
        )
        self.density_plane, self.density_line_time = self.up_sampling_planes(
            self.density_plane, self.density_line_time, res_target, time_grid
        )

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")
