import torch
import torch.nn as nn
import torch.nn.functional as F


class MICELossSegmentation(nn.Module):
    """
    MICE Loss for semantic segmentation.

    这个版本更贴合原论文 MICE 的思想：

      原始 MICE:
        MICE = (A - A0) / (1 - A0)

      损失化后:
        A      -> A_soft = mean P(y_i | x_i)
        A0     -> sum_c p_c^2
        Loss   -> 1 - MICE

    其中：
      A_soft 表示所有有效像素上，模型分配给真实类别的概率平均值；
      A0 表示随机分类器在类别先验分布下的期望正确率。

    支持：
      mice_mode = "fixed": 使用训练集 class_priors
      mice_mode = "batch": 使用当前 batch 的类别分布
    """

    def __init__(
        self,
        num_classes,
        class_priors=None,
        mice_mode="fixed",
        smooth=1e-6,
        ignore_index=255,
        upper_clamp_for_loss=True,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.mice_mode = str(mice_mode).lower()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.upper_clamp_for_loss = bool(upper_clamp_for_loss)

        if self.mice_mode not in ["fixed", "batch"]:
            raise ValueError(f"Unsupported mice_mode: {self.mice_mode}")

        if class_priors is None:
            priors = torch.ones(self.num_classes, dtype=torch.float32) / self.num_classes
        else:
            priors = torch.as_tensor(class_priors, dtype=torch.float32)
            if priors.numel() != self.num_classes:
                raise ValueError(
                    f"class_priors length={priors.numel()} != num_classes={self.num_classes}"
                )
            priors = priors / priors.sum().clamp_min(self.smooth)

        self.register_buffer("class_priors", priors)
        self.register_buffer("oa0", (priors ** 2).sum())

    def forward(self, logits, targets, return_score=False):
        """
        logits:  [B, C, H, W]
        targets: [B, H, W]
        """
        if logits.dim() != 4:
            raise ValueError(f"logits should be [B, C, H, W], got {logits.shape}")

        if targets.dim() != 3:
            raise ValueError(f"targets should be [B, H, W], got {targets.shape}")

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits channel={logits.shape[1]} != num_classes={self.num_classes}"
            )

        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        probs = F.softmax(logits, dim=1)

        valid = targets != self.ignore_index
        valid = valid & (targets >= 0) & (targets < self.num_classes)

        if valid.sum() == 0:
            zero = logits.sum() * 0.0
            if return_score:
                return zero, zero.detach()
            return zero

        probs_flat = probs.permute(0, 2, 3, 1)[valid]  # [N, C]
        targets_flat = targets[valid].long()           # [N]

        correct_probs = probs_flat.gather(
            dim=1,
            index=targets_flat.unsqueeze(1),
        ).squeeze(1)

        # A_soft：可微的 soft accuracy
        soft_acc = correct_probs.mean()

        # A0：随机分类基线
        if self.mice_mode == "batch":
            counts = torch.bincount(
                targets_flat,
                minlength=self.num_classes,
            ).float().to(logits.device)
            priors = counts / counts.sum().clamp_min(self.smooth)
            oa0 = (priors ** 2).sum()
        else:
            oa0 = self.oa0.to(logits.device)

        raw_score = (soft_acc - oa0) / (1.0 - oa0 + self.smooth)

        # 更贴合原论文时，可以只限制上界，避免 perfect 情况超过 1。
        # 不做 lower clamp，避免极端不平衡时 raw_score << -1 后梯度被截断。
        if self.upper_clamp_for_loss:
            upper = torch.tensor(
                1.0 - self.smooth,
                device=raw_score.device,
                dtype=raw_score.dtype,
            )
            score_for_loss = torch.minimum(raw_score, upper)
        else:
            score_for_loss = raw_score

        loss = 1.0 - score_for_loss

        if return_score:
            return loss, raw_score.detach()

        return loss


class CEMICECombinedSegLoss(nn.Module):
    """
    CE / MICE / CE+MICE 统一损失。

    loss_name:
      - ce
      - mice
      - ce_mice

    这里的 MICE 使用更贴合原论文的 map-level MICE：
      A_soft = mean P(y_i | x_i)
      A0 = sum_c p_c^2
      MICE = (A_soft - A0) / (1 - A0)
      Loss = 1 - MICE
    """

    def __init__(
        self,
        num_classes,
        loss_name="ce_mice",
        class_priors=None,
        lambda_mice=0.1,
        smooth=1e-6,
        mice_mode="fixed",
        mice_level="map",
        include_background=True,
        background_index=0,
        ignore_index=255,
        upper_clamp_for_loss=True,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.loss_name = str(loss_name).lower()
        self.lambda_mice = float(lambda_mice)
        self.ignore_index = ignore_index

        if self.loss_name not in ["ce", "mice", "ce_mice"]:
            raise ValueError(f"Unsupported loss_name: {self.loss_name}")

        ce_ignore_index = -100 if ignore_index is None else int(ignore_index)
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)

        # 为了兼容你原来的 config，这里保留 mice_level/include_background/background_index 参数，
        # 但实际不再使用 class-level，只使用 map-level MICE。
        if str(mice_level).lower() != "map":
            print(
                f"[Warning] 当前版本为了贴合原论文，只使用 map-level MICE。"
                f" 你传入的 mice_level={mice_level} 将被忽略。"
            )

        self.mice = MICELossSegmentation(
            num_classes=num_classes,
            class_priors=class_priors,
            mice_mode=mice_mode,
            smooth=smooth,
            ignore_index=ignore_index,
            upper_clamp_for_loss=upper_clamp_for_loss,
        )

        print(f"[Loss] loss_name={self.loss_name}")
        print(f"[Loss] num_classes={self.num_classes}")
        print(f"[Loss] ignore_index={self.ignore_index}")
        print(f"[Loss] lambda_mice={self.lambda_mice}")
        print(f"[Loss] mice_mode={mice_mode}")
        print(f"[Loss] mice_level=map")
        print(f"[Loss] smooth={smooth}")
        print(f"[Loss] upper_clamp_for_loss={upper_clamp_for_loss}")
        print(f"[Loss] class_priors={class_priors}")

    def forward(self, logits, targets, return_details=False):
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        ce_loss = self.ce(logits, targets)
        mice_loss, mice_score = self.mice(logits, targets, return_score=True)

        if self.loss_name == "ce":
            total_loss = ce_loss
        elif self.loss_name == "mice":
            total_loss = mice_loss
        else:
            total_loss = ce_loss + self.lambda_mice * mice_loss

        if return_details:
            return {
                "loss": total_loss,
                "ce_loss": ce_loss,
                "mice_loss": mice_loss,
                "mice_score": mice_score,
            }

        return total_loss
    



class ClassWiseMICELossSegmentation(nn.Module):
    """
    Class-wise MICE Loss for semantic segmentation.

    思路：
      对每个类别 c 单独计算：
        RA_soft_c = mean P(c | x_i), where y_i = c
        A0_c = prior_c
        MICE_c = (RA_soft_c - A0_c) / (1 - A0_c)
        Loss_c = 1 - MICE_c

      最后对类别求平均，而不是对像素求平均。
      这样少数类不会被背景像素淹没。
    """

    def __init__(
        self,
        num_classes,
        class_priors=None,
        mice_mode="fixed",
        smooth=1e-6,
        ignore_index=255,
        include_background=True,
        background_index=0,
        upper_clamp_for_loss=True,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.mice_mode = str(mice_mode).lower()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.include_background = bool(include_background)
        self.background_index = int(background_index)
        self.upper_clamp_for_loss = bool(upper_clamp_for_loss)

        if self.mice_mode not in ["fixed", "batch"]:
            raise ValueError(f"Unsupported mice_mode: {self.mice_mode}")

        if class_priors is None:
            priors = torch.ones(self.num_classes, dtype=torch.float32) / self.num_classes
        else:
            priors = torch.as_tensor(class_priors, dtype=torch.float32)
            priors = priors / priors.sum().clamp_min(self.smooth)

        self.register_buffer("class_priors", priors)

    def forward(self, logits, targets, return_score=False):
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        probs = F.softmax(logits, dim=1)

        valid = targets != self.ignore_index
        valid = valid & (targets >= 0) & (targets < self.num_classes)

        if valid.sum() == 0:
            zero = logits.sum() * 0.0
            if return_score:
                return zero, zero.detach()
            return zero

        probs_flat = probs.permute(0, 2, 3, 1)[valid]  # [N, C]
        targets_flat = targets[valid].long()           # [N]

        if self.mice_mode == "batch":
            counts = torch.bincount(
                targets_flat,
                minlength=self.num_classes,
            ).float().to(logits.device)
            priors = counts / counts.sum().clamp_min(self.smooth)
        else:
            priors = self.class_priors.to(logits.device)

        class_scores = []

        for c in range(self.num_classes):
            if (not self.include_background) and c == self.background_index:
                continue

            cls_mask = targets_flat == c

            # 当前 batch 中没有这个类，就跳过，避免无意义梯度
            if cls_mask.sum() == 0:
                continue

            # 类别 c 的 soft recall / reference accuracy
            ra_soft_c = probs_flat[cls_mask, c].mean()

            # 原论文 class-level efficacy 的随机基线
            a0_c = priors[c].clamp(max=1.0 - self.smooth)

            score_c = (ra_soft_c - a0_c) / (1.0 - a0_c + self.smooth)

            if self.upper_clamp_for_loss:
                upper = torch.tensor(
                    1.0 - self.smooth,
                    device=score_c.device,
                    dtype=score_c.dtype,
                )
                score_c = torch.minimum(score_c, upper)

            class_scores.append(score_c)

        if len(class_scores) == 0:
            zero = logits.sum() * 0.0
            if return_score:
                return zero, zero.detach()
            return zero

        class_score = torch.stack(class_scores).mean()
        loss = 1.0 - class_score

        if return_score:
            return loss, class_score.detach()

        return loss
    
class CEMICEAblationSegLoss(nn.Module):
    """
    用于消融实验的 CE / Map-wise MICE / Class-wise MICE 融合模块。

    不替换你原来的 CEMICECombinedSegLoss。
    这个模块专门用于比较：

      1. CE
      2. CE + Map-wise MICE
      3. CE + Class-wise MICE
      4. CE + Map-wise MICE + Class-wise MICE

    loss_name 支持：
      - "ce"
      - "ce_map_mice"
      - "ce_class_mice"
      - "ce_map_class_mice"

    可选纯 MICE 模式：
      - "map_mice"
      - "class_mice"
      - "map_class_mice"

    推荐主消融使用：
      - ce
      - ce_map_mice
      - ce_class_mice
      - ce_map_class_mice
    """

    def __init__(
        self,
        num_classes,
        loss_name="ce_map_class_mice",
        class_priors=None,
        lambda_map=0.1,
        lambda_cls=0.1,
        smooth=1e-6,
        mice_mode="fixed",
        include_background=False,
        background_index=0,
        ignore_index=255,
        upper_clamp_for_loss=True,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.loss_name = str(loss_name).lower()
        self.lambda_map = float(lambda_map)
        self.lambda_cls = float(lambda_cls)
        self.smooth = float(smooth)
        self.mice_mode = str(mice_mode).lower()
        self.include_background = bool(include_background)
        self.background_index = int(background_index)
        self.ignore_index = ignore_index
        self.upper_clamp_for_loss = bool(upper_clamp_for_loss)

        # 一些别名，方便 config 写法
        alias = {
            "ce_mice": "ce_map_mice",
            "ce_map": "ce_map_mice",
            "ce_class": "ce_class_mice",
            "ce_all": "ce_map_class_mice",
            "ce_map_cls_mice": "ce_map_class_mice",
            "ce_map_class": "ce_map_class_mice",
            "map": "map_mice",
            "class": "class_mice",
            "map_class": "map_class_mice",
        }

        self.loss_name = alias.get(self.loss_name, self.loss_name)

        valid_loss_names = [
            "ce",
            "ce_map_mice",
            "ce_class_mice",
            "ce_map_class_mice",
            "map_mice",
            "class_mice",
            "map_class_mice",
        ]

        if self.loss_name not in valid_loss_names:
            raise ValueError(
                f"Unsupported loss_name={self.loss_name}. "
                f"Supported: {valid_loss_names}"
            )

        # 判断当前模式需要哪些损失
        self.use_ce = self.loss_name in [
            "ce",
            "ce_map_mice",
            "ce_class_mice",
            "ce_map_class_mice",
        ]

        self.use_map_mice = self.loss_name in [
            "ce_map_mice",
            "ce_map_class_mice",
            "map_mice",
            "map_class_mice",
        ]

        self.use_class_mice = self.loss_name in [
            "ce_class_mice",
            "ce_map_class_mice",
            "class_mice",
            "map_class_mice",
        ]

        # CE 损失
        ce_ignore_index = -100 if ignore_index is None else int(ignore_index)
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)

        # Map-wise MICE：你原来的 MICELossSegmentation
        if self.use_map_mice:
            self.map_mice = MICELossSegmentation(
                num_classes=num_classes,
                class_priors=class_priors,
                mice_mode=mice_mode,
                smooth=smooth,
                ignore_index=ignore_index,
                upper_clamp_for_loss=upper_clamp_for_loss,
            )
        else:
            self.map_mice = None

        # Class-wise MICE：你新增的 ClassWiseMICELossSegmentation
        if self.use_class_mice:
            self.class_mice = ClassWiseMICELossSegmentation(
                num_classes=num_classes,
                class_priors=class_priors,
                mice_mode=mice_mode,
                smooth=smooth,
                ignore_index=ignore_index,
                include_background=include_background,
                background_index=background_index,
                upper_clamp_for_loss=upper_clamp_for_loss,
            )
        else:
            self.class_mice = None

        print("[Loss] CEMICEAblationSegLoss")
        print(f"[Loss] loss_name={self.loss_name}")
        print(f"[Loss] num_classes={self.num_classes}")
        print(f"[Loss] ignore_index={self.ignore_index}")
        print(f"[Loss] mice_mode={self.mice_mode}")
        print(f"[Loss] smooth={self.smooth}")
        print(f"[Loss] lambda_map={self.lambda_map}")
        print(f"[Loss] lambda_cls={self.lambda_cls}")
        print(f"[Loss] use_ce={self.use_ce}")
        print(f"[Loss] use_map_mice={self.use_map_mice}")
        print(f"[Loss] use_class_mice={self.use_class_mice}")
        print(f"[Loss] include_background={self.include_background}")
        print(f"[Loss] background_index={self.background_index}")
        print(f"[Loss] upper_clamp_for_loss={self.upper_clamp_for_loss}")
        print(f"[Loss] class_priors={class_priors}")

    def forward(self, logits, targets, return_details=False):
        """
        logits:  [B, C, H, W]
        targets: [B, H, W]
        """

        if logits.dim() != 4:
            raise ValueError(f"logits should be [B, C, H, W], got {logits.shape}")

        if targets.dim() != 3:
            raise ValueError(f"targets should be [B, H, W], got {targets.shape}")

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits channel={logits.shape[1]} != num_classes={self.num_classes}"
            )

        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        zero = logits.sum() * 0.0

        # 1. CE
        if self.use_ce:
            ce_loss = self.ce(logits, targets)
        else:
            ce_loss = zero

        # 2. Map-wise MICE
        if self.use_map_mice:
            map_mice_loss, map_mice_score = self.map_mice(
                logits,
                targets,
                return_score=True,
            )
        else:
            map_mice_loss = zero
            map_mice_score = zero.detach()

        # 3. Class-wise MICE
        if self.use_class_mice:
            class_mice_loss, class_mice_score = self.class_mice(
                logits,
                targets,
                return_score=True,
            )
        else:
            class_mice_loss = zero
            class_mice_score = zero.detach()

        # 4. 总损失
        if self.loss_name == "ce":
            total_loss = ce_loss

        elif self.loss_name == "ce_map_mice":
            total_loss = ce_loss + self.lambda_map * map_mice_loss

        elif self.loss_name == "ce_class_mice":
            total_loss = ce_loss + self.lambda_cls * class_mice_loss

        elif self.loss_name == "ce_map_class_mice":
            total_loss = (
                ce_loss
                + self.lambda_map * map_mice_loss
                + self.lambda_cls * class_mice_loss
            )

        elif self.loss_name == "map_mice":
            total_loss = map_mice_loss

        elif self.loss_name == "class_mice":
            total_loss = class_mice_loss

        elif self.loss_name == "map_class_mice":
            total_loss = (
                self.lambda_map * map_mice_loss
                + self.lambda_cls * class_mice_loss
            )

        else:
            raise RuntimeError(f"Unexpected loss_name={self.loss_name}")

        if return_details:
            return {
                "loss": total_loss,

                # 原始 CE
                "ce_loss": ce_loss.detach(),

                # map-wise MICE
                "map_mice_loss": map_mice_loss.detach(),
                "map_mice_score": map_mice_score.detach(),

                # class-wise MICE
                "class_mice_loss": class_mice_loss.detach(),
                "class_mice_score": class_mice_score.detach(),

                # 兼容旧日志字段
                "mice_loss": (map_mice_loss + class_mice_loss).detach(),
                "mice_score": map_mice_score.detach()
                if self.use_map_mice
                else class_mice_score.detach(),

                # 当前权重
                "lambda_map": self.lambda_map,
                "lambda_cls": self.lambda_cls,
            }

        return total_loss
    

class SoftAccuracyLoss(nn.Module):
    """
    SoftAcc = mean p(y_i | x_i)
    loss = 1 - SoftAcc

    注意：
    - 这里是不带 A0 归一化的 raw soft accuracy。
    - CE+Map-MICE 等价于带 A0 归一化的 SoftAcc regularization。
    """

    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        probs = torch.softmax(logits, dim=1)

        valid = target != self.ignore_index
        valid = valid & (target >= 0) & (target < logits.shape[1])

        if valid.sum() == 0:
            zero = logits.sum() * 0.0
            return zero, zero.detach()

        safe_target = target.clone()
        safe_target[~valid] = 0

        p_true = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        soft_acc = p_true[valid].mean()

        loss = 1.0 - soft_acc

        return loss, soft_acc.detach()


class CESoftAccLoss(nn.Module):
    def __init__(
        self,
        ignore_index=255,
        lambda_softacc=0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_softacc = float(lambda_softacc)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.softacc = SoftAccuracyLoss(ignore_index=ignore_index, eps=eps)

    def forward(self, logits, target):
        if isinstance(logits, (list, tuple)):
            main_logits = logits[0]
        else:
            main_logits = logits

        ce_loss = self.ce(main_logits, target)
        softacc_loss, softacc_score = self.softacc(main_logits, target)

        total = ce_loss + self.lambda_softacc * softacc_loss

        details = {
            "ce": float(ce_loss.detach().cpu()),
            "softacc_loss": float(softacc_loss.detach().cpu()),
            "softacc_score": float(softacc_score.detach().cpu()),
            "mice_loss": 0.0,
            "mice_score": 0.0,
        }

        return total, details