# 拜占庭鲁棒、投影压缩的自适应联邦学习：算法、假设与收敛性分析

> **本文档是「变形 A + 客户端动量」版本**，由 `算法与收敛性分析_变形A.md` 修改而来。
> 相对变形A 有**一处算法改动**（§2.2 第 1–2 步）：每个诚实客户端维护一个动量缓存，EF21 的 correction 针对缓存而非原始随机梯度构造。
>
> **动机**：变形A（以及原完整版）的 error floor 中有一项正比于 $\sigma^2/(Hc^2)$，它**与步长无关、不随 $T$ 消失，且即使不压缩（$p=1$）也保留 $O(\sigma^2/H)$ 的常数**。根源是 EF21 的驱动项 $\mathbb E\|\bar g_t-\bar g_{t-1}\|^2$ 含两轮独立采样噪声（引理 11.9 的讨论）。服务器侧的 Adam 动量作用在聚合输出上，治不了这个位于 tracker **输入端**的问题。
>
> **结果**：客户端动量使 error floor 中**全部含 $\sigma^2$ 的项获得因子 $(1-\beta)$ 或 $(1-\beta)^2$**（推论 12.3）。取 $1-\beta=\Theta(T^{-1/2})$ 后，噪声 floor 以 $O(T^{-1/2})$ 消失，**最终 error floor 只含异质性 $\zeta^2$**（推论 12.4）——而 $\zeta^2$ 部分有匹配下界，本就不可消除。与已知下界的全部差距只剩一个压缩代价因子 $1/p$（注 12.6）。
>
> **代价**：步长条件新增 $\eta\lesssim1-\beta$（注 11.16），以及一个新的动量偏差项 $\mathcal B_t$ 需要闭合（§11.3）。假设集合**不变**：仍只用 A1–A4 与抽象结构假设 A5–A7，**不使用任何有界（随机）梯度假设**。
>
> **继承自变形A 的两处改动**（本版保留）：
> 1. 裁剪改为**预聚合 ARC**（Allouah et al. 2025），被吸收进 $(f,\kappa)$-robustness（定理 10.4），不产生任何残差；
> 2. cap **只作用于二阶矩累加器**，一阶矩用未裁剪的 $z_t$，Adam 的 preconditioner 谱界（引理 11.1）因此原封不动而更新方向无裁剪偏差。
>
> 通信量、通信轮次、压缩率与变形A 完全一致；客户端每轮多存一个 $d$ 维缓存、多一次向量线性组合。
> 完整的逐项对照见 §12.4 与文末改动清单。

## 文档组织与阅读方式

全文分两个层次，这一分层是刻意的：

* **第 I 部分（§1–§13）：抽象分析。**
 两个 Byzantine-robust 模块与 Top-K 模块只以**抽象性质**出现——
 $(f,\kappa_r,s)$-robust subset（A5）、$(f,\kappa_q)$-robust aggregation（A6）、
 $(\omega_{\mathrm{top}},\delta_{\mathrm{JL}})$-projected Top-K contraction（A7）。
 主定理（定理 12.1 / 12.2）只用到这三条性质与常规光滑/方差假设，
 不依赖任何具体的聚合规则或投影分析。
* **第 II 部分（§14–§16）：可实现性（instantiation）。**
 单独证明这三条抽象假设**不是空假设**：本算法的
 "投影空间 Multi-Krum + 单套 Gaussian 投影 + projected Top-K"
 组合可以同时满足 A5、A6、A7，并给出 $\kappa_r,\kappa_q,\omega_{\mathrm{top}},\delta_{\mathrm{JL}}$
 的显式表达式（定理 14.8、命题 14.12、定理 14.10、推论 14.2）。
 §14.9 把这些常数回代主定理，得到端到端的实例化推论（推论 14.15）。

这样安排的理由：主定理的正确性与"选哪个聚合器"解耦，便于替换模块；
而第 II 部分保证理论**落地**——存在一个可实际执行、通信量符合 (2.11) 的实现使全部假设成立。

---

## 0. 本文相对原草稿的改动与新增结果一览

原草稿第三部分列出 10 项待解决问题，下表逐项对应。

| 原草稿待解决问题 | 本文处理 | 结论 |
|---|---|---|
| §1 主定理未用到 $\kappa_r$ 与 $\omega_{\mathrm{top}}$ | 定理 9.5、定理 12.2 | 新建 $\omega_{\mathrm{top}}$-递推与 $p$-递推并列，主定理 B 取二者较优，四个模块全部进入速率 |
| §2.1 correction-side selector 未实例化 | 命题 14.5、引理 14.7、定理 14.8 | **Multi-Krum 满足 A5**：其 score 只依赖成对距离，与 JL 天然相容；$\kappa_r$ 显式 |
| §2.2 tracker-side aggregator 未实例化 | 命题 14.12 | 复用 Multi-Krum（全维、无 JL 依赖），$\kappa_q$ 显式；§14.7 给出几何中位数/CWTM/NNM 的替代与加强 |
| §3 JL good event 与失败概率 | A7 的形式 + 引理 9.3、注 9.4 | **不必退化为 high-probability 定理**：可保留无条件期望界，且 $\log\frac1{\delta_{\mathrm{JL}}}$ 中**不含 $T$** |
| §4 clipping residual 未闭合 | **§10 全新（ARC）+ §11.7** | **彻底解决**：裁剪改为预聚合并被 $(f,\kappa)$-robustness 吸收（定理 10.4），cap 移到二阶矩路径，残差项**不再存在**（注 11.8）。注 11.8a 与注 3.2 证明完整版那条路在只有 A3 时必然不可闭合 |
| §5 momentum residual 未闭合 | 定理 11.9 | 采用方案 B：$\bar R_{\mathrm{mom}}^{(T)}$ 被 $\bar\Delta_{\mathrm{est}}^{(T)}$ 与 $\eta^2$ 完全吸收，主定理不再残留该项 |
| §6 filtration 未统一 | §5、(M1)–(M4) | 给出单轮六段式 filtration，所有条件期望重述 |
| §7 $\rho$ 与 $\omega_{\mathrm{top}}$ 权衡 | 推论 13.1、§14.9.1 | 给出 $\rho^\star$ 所满足的方程与两个主定理的占优区间 |
| §8 投影维数 $\log\binom Ns$ 过保守 | 推论 14.2 | 改用 subspace embedding：$r\gtrsim\varepsilon_{\mathrm{JL}}^{-2}(\min\{N,n\}+\log\frac m{\delta_{\mathrm{JL}}})$，**一个事件同时给出 A5 与 A7** |
| §9 定理分层 | 定理 12.1、定理 12.2 | 定理 A（基础版，仅 refresh + Agg-$q$）与定理 B（完整算法版） |

此外本文修正了原草稿三处不必要或不严密之处：

* **删除多余的协议条件**：原 Condition 2（globally-consistent mask）与 Condition 3（uniform random refresh）不是假设，而是 Algorithm 1 第 7 步的**构造事实**（服务器生成并广播唯一的 $I_t$），已并入算法描述；只保留真正约束攻击者行为空间、需要密码学手段落实的 Condition 1（承诺机制），并在 §4 给出它**必要性**的具体攻击构造。
* **不引入有界梯度假设，且主定理完全闭合**：定理 12.1/12.2 **不使用**任何形式的有界梯度/有界随机梯度假设，右端也**不含任何未界定的量**。Adam 所需的 preconditioner 界由「只作用于二阶矩的 cap」**构造性地**保证（引理 11.1）；裁剪偏差因 ARC 的预聚合结构而**根本不出现**（注 11.8）。注 11.8a 给出一个反例，说明完整版那条路（后聚合 cap + 显式残差）在只有有界方差时**必然**无法闭合，因此这里的改动不是优化而是必要。
* **修正参数定义与边界情形**：原递推 (41) 只对 $t\ge1$ 成立，原文的时间平均未单独处理 $E_0$ 与无定义的 $D_0$，本文在引理 9.5 中显式处理。
* **（变形 A 的更正）** 完整版曾把原草稿的 $k_{\mathrm{clip}}=\lfloor2\alpha(1-\alpha)N\rfloor$ 判为「$\alpha$ 未定义、算法无从计算」并放宽为区间取值。这是误读：该式即 ARC 的 (2.0)，其中的比例是**已知的容忍参数** $\hat B/N$。本版恢复原草稿的取法，因为引理 10.2 表明它是定理 10.4 与引理 10.8 的必要前提（注 2.1）。

**本文的分析完整覆盖 non-i.i.d. 设定**：异质性 $\zeta^2$（A4）显式贯穿 $V_{g,t}\to V_{q,t}\to V_{r,t}\to$ error floor，且 §15.2 说明其系数与已知下界 $\Omega(\tfrac BN\zeta^2)$ 的差距。

---

# 第 I 部分：抽象分析

## 1. 记号与问题设定

### 1.1 系统与目标

$N$ 个客户端，honest 集合 $\mathcal H$（$H:=|\mathcal H|$），Byzantine 集合 $\mathcal B$（$B:=|\mathcal B|$），$N=H+B$。$\mathcal B$ 固定但对算法未知。

$$
f(\theta)=\frac1H\sum_{i\in\mathcal H}f_i(\theta),
\qquad
f_i(\theta)=\mathbb E_{\zeta^{(i)}}\big[\ell(\theta;\zeta^{(i)})\big],
\qquad
h_t:=\nabla f(\theta_t).
\tag{1.1}
$$

### 1.2 矩阵化

固定 reshape $d=mn$，$x\in\mathbb R^d\leftrightarrow X\in\mathbb R^{m\times n}$，第 $j$ 行记 $X_{j,:}\in\mathbb R^n$，于是 $\|x\|^2=\sum_{j=1}^m\|X_{j,:}\|^2$。

### 1.3 核心量

| 记号 | 定义 | 含义 |
|---|---|---|
| $g_t^{(i)}$ | $\nabla\ell(\theta_t;\zeta_t^{(i)})$ | honest 随机梯度 |
| $u_t^{(i)}$ | 递推 (2.1a) | 客户端动量缓存（**本版新增**；tracker 追踪的对象） |
| $\bar u_t$ | $\frac1H\sum_{i\in\mathcal H}u_t^{(i)}$ | honest 动量均值 |
| $V_{u,t}$ | $\frac1H\sum_{i\in\mathcal H}\|u_t^{(i)}-\bar u_t\|^2$ | 动量缓存的 honest dispersion |
| $\mathcal B_t$ | $\mathbb E\|\bar u_t-h_t\|^2$ | **客户端动量偏差**（本版新增，§11.3） |
| $\Sigma_\beta^2$ | $2\zeta^2+\frac{2(1-\beta)}{1+\beta}\sigma^2$ | 动量后的 dispersion 界（(7.4)），取代变形A 的 $\Sigma^2$ |
| $\beta$ | 客户端动量参数 | 与服务器 Adam 的 $\beta_1$ **不同**；$c_\beta^{\mathrm{cl}}=\beta^2/(1-\beta)^2$ |
| $q_t^{(i)}$ | 递推 (2.6) | 客户端 $i$ 的 EF21 tracker |
| $r_t^{(i)}$ | $u_t^{(i)}-q_{t-1}^{(i)}$ | EF21 correction（**本版针对动量缓存**） |
| $\bar g_t,\bar q_t,\bar r_t$ | $\frac1H\sum_{i\in\mathcal H}(\cdot)$ | honest 均值 |
| $\bar r_t^S$ | $\frac1s\sum_{i\in\mathcal S_t}r_t^{(i)}$ | selected-subset correction 均值（**可能含 Byzantine**） |
| $\Delta_t$ | $\bar r_t^S-\bar r_t$ | correction-side robust subset 偏差 |
| $V_{g,t},V_{q,t},V_{r,t}$ | $\frac1H\sum_{i\in\mathcal H}\|\cdot^{(i)}-\overline{\cdot}_t\|^2$ | honest dispersion |
| $a_t^{(i)},b_t^{(i)}$ | $q_t^{(i)}-\bar q_t$, $u_t^{(i)}-\bar u_t$ | 中心化量（**本版 $b$ 针对 $u$**） |
| $e_t,\;E_t$ | $\bar q_t-\bar u_t$, $\mathbb E\|e_t\|^2$ | honest 均值的 EF21 tracking error（**针对 $\bar u$**） |
| $D_t$ | $\mathbb E\|\bar u_t-\bar u_{t-1}\|^2$ | honest 均值**动量**漂移（本版收益的所在，引理 11.9） |
| $\Sigma^2$ | $\sigma^2+\zeta^2$ | |
| $p$ | $k_{\mathrm{rnd}}/(m-k_{\mathrm{top}})$ | 单行 refresh 概率 |
| $\gamma,\rho$ | $k/m$, $k_{\mathrm{rnd}}/k$ | 总行压缩比、refresh 占比 |

---

## 2. 算法

### 2.1 参数

客户端：动量参数 $\beta\in[0,1)$（**本版新增**）。

服务器：$\eta>0$，Adam 参数 $\beta_1,\beta_2\in[0,1)$，$\epsilon>0$，cap $C_{\max}>0$。ARC 的裁剪计数**不是自由参数**，由已知的 Byzantine 上界 $\hat B$（A8）确定：

$$
k_{\mathrm{arc}}=\Big\lfloor 2\tfrac{\hat B}N\big(N-\hat B\big)\Big\rfloor ,
\tag{2.0}
$$

唯一要求是 A8 已经给出的 $N>2\hat B$。

> **注 2.1（相对原版的修正）** 原版把裁剪计数放宽为「$\hat B\le k_{\mathrm{clip}}<H$ 内任取」，并批评原草稿的取法 $\lfloor2\alpha(1-\alpha)N\rfloor$「$\alpha$ 未定义、算法无从计算」。该批评是误读：(2.0) 中的比例是**容忍参数** $\hat B/N$（算法已知），不是真实的 $B/N$。而且这个特定取法是必需的——引理 10.2 表明它同时保证「至少 $\hat B+1$ 个输入范数不低于阈值」（阈值不塌陷）与「任意大小 $N-\hat B$ 的子集中必有一个输入范数不低于阈值」（输出有界性），二者分别是定理 10.4 与引理 10.8 的前提。因此本版恢复原草稿的取法。

> **注 2.2（cap 的唯一职责）** 本版中 $C_{\max}$ **只用于二阶矩累加器**（见第 12 步），不作用于更新方向。它的唯一作用是给 Adam 的 preconditioner 一个确定性谱界（引理 11.1）。这是变形 A 与原版最本质的差别：原版让 $C_{\max}$ 同时承担「界住 preconditioner」与「界住更新方向」两件事，后者产生了无法用问题参数闭合的 clipping bias（注 11.8a 证明它在只有 A3 时**必然**无法闭合）。
压缩：reshape $(m,n)$，投影维数 $r$，$k=k_{\mathrm{top}}+k_{\mathrm{rnd}}$，要求

$$
1\le k_{\mathrm{top}}<m,
\qquad
1\le k_{\mathrm{rnd}}\le m-k_{\mathrm{top}}.
\tag{2.1}
$$

鲁棒模块：**抽象**算子 $\mathcal R_r$（correction-side subset selector，输出大小 $s$）与 $\mathcal A_q$（tracker-side aggregator）。第 I 部分只要求它们满足 A5/A6；§14 给出 Multi-Krum 实例化。

### 2.2 伪代码

> **Algorithm 1**（RP-EF21-Adam：Randomly-Projected EF21 with Two-Sided Robustness and Capped Adaptive Clipping）

**输入**：$\theta_0$；$m_0=0$，$\tilde v_0=0$；$q_{-1}^{(i)}=0$ 及服务器副本 $q_{-1,\mathrm{srv}}^{(i)}=0$（$i\in[N]$）。

对 $t=0,1,\ldots,T-1$：

1. **（客户端）随机梯度与动量缓存**：honest $i\in\mathcal H$ 采样 $\zeta_t^{(i)}$，$g_t^{(i)}=\nabla\ell(\theta_t;\zeta_t^{(i)})$，并更新本地动量缓存
$$
u_0^{(i)}=g_0^{(i)},
\qquad
u_t^{(i)}=\beta\,u_{t-1}^{(i)}+(1-\beta)\,g_t^{(i)}\quad(t\ge1).
\tag{2.1a}
$$
 *这是本版相对变形A 的**唯一**算法改动。缓存是纯本地状态，不上传、不参与通信；$\beta=0$ 时退化为变形A。初值取 $u_0=g_0$（而非 $0$）以避免预热偏差，见引理 11.4 的证明。*
2. **（客户端）correction 承诺**：
$$
r_t^{(i)}=u_t^{(i)}-q_{t-1}^{(i)},
\qquad r_t^{(i)}\leftrightarrow R_t^{(i)}\in\mathbb R^{m\times n},
\tag{2.2}
$$
 此刻 $r_t^{(i)}$ 对每个客户端（含 Byzantine）**必须被承诺**（Condition 1）。
3. **（服务器）公布投影**：抽 $V_t\in\mathbb R^{n\times r}$，$(V_t)_{ab}\stackrel{\text{iid}}\sim\mathcal N(0,1)$，广播 seed。
4. **（客户端）第一阶段上传**：
$$
P_t^{(i)}=\tfrac1{\sqrt r}R_t^{(i)}V_t\in\mathbb R^{m\times r}
\qquad(\text{每客户端 }mr\text{ 个浮点数}).
\tag{2.3}
$$
5. **（服务器）投影空间鲁棒子集选择**：
$$
\mathcal S_t=\mathcal R_r\big(P_t^{(1)},\ldots,P_t^{(N)}\big),
\quad|\mathcal S_t|=s;
\qquad
\bar P_t^S=\tfrac1s\!\!\sum_{i\in\mathcal S_t}\!\!P_t^{(i)}=\tfrac1{\sqrt r}\bar R_t^SV_t .
\tag{2.4}
$$
6. **（服务器）row score 与 Top-$k_{\mathrm{top}}$**：
$$
\sigma_{t,j}=\big\|(\bar P_t^S)_{j,:}\big\|^2\ (j\in[m]),
\qquad
I_t^{\mathrm{top}}=\mathrm{TopK}(\sigma_t,k_{\mathrm{top}}).
\tag{2.5}
$$
7. **（服务器）uniform random refresh**：从 $\mathcal C_t=[m]\setminus I_t^{\mathrm{top}}$ 均匀无放回抽 $k_{\mathrm{rnd}}$ 行得 $I_t^{\mathrm{rnd}}$；$I_t=I_t^{\mathrm{top}}\cup I_t^{\mathrm{rnd}}$（两集合不交，故 $|I_t|=k$），广播 $I_t$。$Q_t$ 为保留 $I_t$ 行、其余置零的正交投影，$Q_t^{\mathrm{top}}$ 对应 $I_t^{\mathrm{top}}$。
 *由于 $I_t^{\mathrm{rnd}}$ 只被生成一次、$I_t$ 被广播给全体客户端，mask 的全局一致性（$Q_t^{(1)}=\cdots=Q_t^{(N)}=Q_t$）与 refresh 的均匀性都是本步的构造事实，无须另列为假设（注 4.1）。*
8. **（客户端）第二阶段上传**：$c_t^{(i)}=\mathrm{vec}\big(R_t^{(i)}[I_t,:]\big)\in\mathbb R^{kn}$（$kn$ 个浮点数），须满足 $U_tc_t^{(i)}=Q_tr_t^{(i)}$。
9. **（双方）tracker 更新**：对 $i\in\mathcal H$，
$$
q_t^{(i)}=q_{t-1}^{(i)}+U_tc_t^{(i)}
=(I-Q_t)q_{t-1}^{(i)}+Q_tu_t^{(i)} ;
\tag{2.6}
$$
 逐行：$j\in I_t\Rightarrow q_{t,j}^{(i)}=u_{t,j}^{(i)}$，$j\notin I_t\Rightarrow q_{t,j}^{(i)}=q_{t-1,j}^{(i)}$。**tracker 现在追踪的是动量缓存 $u_t^{(i)}$ 而非原始梯度**；(2.6) 的逐行结构与全局一致 mask 完全不变，故 §6–§9 的正交分解论证一字不改（注 7.7）。服务器对全部 $i\in[N]$ 用收到的 $c_t^{(i)}$ 同样更新副本。
10. **（服务器）ARC 预聚合裁剪**：把 $\{\|q_t^{(i)}\|\}_{i\in[N]}$ 降序排为 $\pi_t$，取阈值并对**全部 $N$ 份 tracker 副本**裁剪：
$$
C_t^{\mathrm{arc}}=\big\|q_t^{(\pi_t(k_{\mathrm{arc}}+1))}\big\|,
\qquad
\tilde q_t^{(i)}=q_t^{(i)}\min\Big\{1,\tfrac{C_t^{\mathrm{arc}}}{\|q_t^{(i)}\|}\Big\}
\quad(i\in[N]).
\tag{2.7}
$$
 *这一步只作用于服务器的**临时**副本：客户端自身的 tracker 与服务器用于下一轮 (2.6) 更新的持久副本均保持未裁剪，因此 §6–§9 的全部分析对象不变。*
11. **（服务器）tracker 侧鲁棒聚合**：
$$
z_t=\mathcal A_q\big(\tilde q_t^{(1)},\ldots,\tilde q_t^{(N)}\big)
=\big(\mathcal A_q\circ\mathrm{ARC}\big)\big(q_t^{(1)},\ldots,q_t^{(N)}\big).
\tag{2.8}
$$
 **不再对 $z_t$ 做后置裁剪。**
12. **（服务器）Adam 与模型更新**：令 $\hat g_t:=\mathrm{clip}_{C_{\max}}(z_t)=z_t\min\{1,C_{\max}/\|z_t\|\}$（$z_t=0$ 时取 $0$），则
$$
m_{t+1}=\beta_1m_t+(1-\beta_1)\,\underline{z_t},
\quad
\tilde v_{t+1}=\beta_2\tilde v_t+(1-\beta_2)\,\underline{\hat g_t}^{\odot2},
\quad
v_{t+1}=\tfrac{\tilde v_{t+1}}{1-\beta_2^{t+1}},
\tag{2.9}
$$
 **注意两处下划线：一阶矩用未裁剪的 $z_t$，二阶矩用裁剪后的 $\hat g_t$。** 于是确定性地 $v_{t+1,j}\le C_{\max}^2$（引理 11.1），而更新方向不含任何裁剪偏差。
$$
\theta_{t+1}=\theta_t-\eta\,\frac{m_{t+1}}{\sqrt{v_{t+1}}+\sqrt\epsilon}
\quad(\text{逐坐标}),
\qquad\text{广播 }\theta_{t+1}.
\tag{2.10}
$$

### 2.3 通信量

$$
\rho_{\mathrm{comm}}=\frac{mr+kn}{mn}=\frac rn+\frac km .
\tag{2.11}
$$

random refresh 只把既有的 $k$ 个 row 预算拆成 $k_{\mathrm{top}}+k_{\mathrm{rnd}}$，故**不增加通信量、不增加通信轮次、不引入第二套投影、不需广播完整 tracker**。$r$ 的理论取值见 §14.1 与 §13.4。

---

## 3. 假设

### 3.1 标准假设（含 non-i.i.d.）

**A1（光滑性与下有界）** 每个 $f_i$（$i\in\mathcal H$）与 $f$ 均为 $L$-smooth；$f\ge f^\star>-\infty$，记 $\Delta_0=f(\theta_0)-f^\star$。

**A2（无偏性）** $\mathbb E[g_t^{(i)}\mid\mathcal F_t]=\nabla f_i(\theta_t)$，$i\in\mathcal H$。

**A3（有界方差与独立性）** $\mathbb E[\|g_t^{(i)}-\nabla f_i(\theta_t)\|^2\mid\mathcal F_t]\le\sigma^2$；不同 honest 客户端的 mini-batch 噪声条件独立，每轮 fresh 采样。故

$$
\mathbb E\big[\|\bar g_t-h_t\|^2\mid\mathcal F_t\big]\le\frac{\sigma^2}H .
\tag{3.1}
$$

**A4（有界异质性 / non-i.i.d.）** 对任意 $\theta$，

$$
\frac1H\sum_{i\in\mathcal H}\big\|\nabla f_i(\theta)-\nabla f(\theta)\big\|^2\le\zeta^2 .
\tag{3.2}
$$

由 A2–A4，

$$
\mathbb E\big[V_{g,t}\mid\mathcal F_t\big]
=\mathbb E\Big[\tfrac1H\!\!\sum_{i\in\mathcal H}\!\|g_t^{(i)}-\bar g_t\|^2\ \Big|\ \mathcal F_t\Big]
\le\sigma^2+\zeta^2=\Sigma^2 .
\tag{3.3}
$$

> **注 3.0（假设仍施加在原始随机梯度上）** A2–A4 全部是关于 $g_t^{(i)}$ 的，与变形A **完全相同**——引入客户端动量**不改变任何假设**。动量缓存 $u_t^{(i)}$ 的性质（dispersion、偏差）不是假设，而是由 A2–A4 **推导**出来的：引理 7.1 给出 $\mathbb EV_{u,t}$ 的界，引理 11.4 给出 $\mathcal B_t=\mathbb E\|\bar u_t-h_t\|^2$ 的递推。
> 特别注意 $u_t^{(i)}$ **不是** $\nabla f_i(\theta_t)$ 的无偏估计，故 A2 对 $u$ 不成立；这正是 §11.3 存在的原因。

> **注 3.1（本文完整覆盖 non-i.i.d. 设定，且这是分析的实质困难所在）**
> A4 是标准的异质性刻画：$\zeta^2=0$ 对应各 honest 客户端同分布（i.i.d.），$\zeta^2>0$ 为一般 non-i.i.d.。
> $\zeta^2$ 并非可被"处理掉"的技术项，而是贯穿全文的核心量：它经
> $$
> \tfrac1T\textstyle\sum_t V_{u,t}\le\Sigma_\beta^2
> \ \Longrightarrow\
> \bar V_q^{(T)}\le\frac{\Sigma_\beta^2}p
> \ \Longrightarrow\
> \bar V_r^{(T)}\le\Big(1+\tfrac1{\sqrt p}\Big)^2\Sigma_\beta^2
> \ \Longrightarrow\
> \text{error floor}=O\Big(\frac{\kappa_q^\star\Sigma_\beta^2}p\Big)
> $$
> 直接进入主定理。这是**必然的**：在 $(f,\zeta^2)$-异质性下，任何 Byzantine-robust 算法的
> 稳态误差都有 $\Omega\big(\tfrac Bn\zeta^2\big)$ 的下界（Karimireddy–He–Jaggi 2022；Allouah et al. 2023），
> 因为鲁棒聚合器在信息论意义上无法区分"诚实但异质"与"拜占庭"的更新。
> **本版的关键点**：$\Sigma_\beta^2=2\zeta^2+\frac{2(1-\beta)}{1+\beta}\sigma^2$ 中，客户端动量把 $\sigma^2$ 压小而**对 $\zeta^2$ 无作用**——这与上述下界完全一致：可压小的本来就只有噪声。取 $1-\beta=\Theta(T^{-1/2})$ 后 error floor 只剩 $O(\kappa_q^\star\zeta^2/p)$（推论 12.4）。
> 因此正确的理论目标不是消去 $\zeta^2$，而是让它的系数尽量接近下界——见 §14.7 与 §15.2：
> Multi-Krum 给 $\kappa=O(1)$，加 NNM 预聚合可降至 $\kappa=O(B/N)$，与下界匹配到一个 $1/p$ 的压缩代价因子。

### 3.2 三条抽象结构假设

这三条是第 I 部分的全部"算法侧"输入。第 II 部分证明它们**可以被本算法同时满足**。

> **A5（$(f,\kappa_r,s)$-robust subset property，correction 侧）**
> 存在常数 $\kappa_r\ge0$ 与事件族 $\{\mathcal E_t\}_{t\ge0}$，$\mathcal E_t\in\mathcal G_t$，使得
> $$
> \mathbb E\Big[\big\|\bar r_t^S-\bar r_t\big\|^2\,\mathbf 1_{\mathcal E_t}\ \Big|\ \mathcal F_t^{c}\Big]
> \ \le\ \kappa_r\,V_{r,t}
> \qquad\text{a.s.}
> \tag{3.4}
> $$

> **A6（$(f,\kappa_q)$-robust aggregation，tracker 侧）**
> 存在常数 $\kappa_q\ge0$ 使得对任意输入 $q_t^{(1)},\ldots,q_t^{(N)}$（Byzantine 分量任意），
> $$
> \big\|z_t-\bar q_t\big\|^2\ \le\ \kappa_q\,V_{q,t}
> \qquad\text{a.s.}
> \tag{3.5}
> $$

> **A7（$(\omega_{\mathrm{top}},\delta_{\mathrm{JL}})$-projected Top-K contraction）**
> 存在 $\omega_{\mathrm{top}}\in(0,1]$、$\delta_{\mathrm{JL}}\in[0,1)$，使得**同一**事件族 $\{\mathcal E_t\}$（与 A5 相同）满足
> $$
> \Pr\big(\mathcal E_t\mid\mathcal F_t^{c}\big)\ \ge\ 1-\delta_{\mathrm{JL}}
> \qquad\text{a.s.},
> \tag{3.6}
> $$
> 且在 $\mathcal E_t$ 上
> $$
> \big\|(I-Q_t^{\mathrm{top}})\,\bar r_t^S\big\|^2
> \ \le\ (1-\omega_{\mathrm{top}})\,\big\|\bar r_t^S\big\|^2 .
> \tag{3.7}
> $$

**关于 A5–A7 的三点说明。**

1. **A5 不要求 $\mathcal S_t\subseteq\mathcal H$。** 它只要求"被选子集的均值代表 honest 均值"，偏差由 honest dispersion $V_{r,t}$ 控制。这正是 $(f,\kappa)$-robustness 的标准形式（Karimireddy et al. 2021；Allouah et al. 2023），只是把输出限定为"某个大小 $s$ 的子集平均"，因此对 Multi-Krum 这类 selection-then-average 规则是自然的。
2. **A5 与 A7 共用事件族 $\mathcal E_t$，且都只在 $\mathcal E_t$ 上成立。** 这是不可避免的：在 $\mathcal E_t$ 之外，Byzantine 的 correction 可任意大，(3.4) 不可能无条件成立。原草稿据此认为主定理必须退化为 high-probability 形式；**§9 的引理 9.3 说明并非如此**——由于 $(I-Q_t^{\mathrm{top}})$ 是压缩映射而 $\bar r_t$ 是纯 honest 量，$\mathcal E_t$ 的补集上根本不会出现 Byzantine 量，只要 $\delta_{\mathrm{JL}}\le\omega_{\mathrm{top}}/4$ 就能保留**无条件期望**主定理，且 $\log\frac1{\delta_{\mathrm{JL}}}$ 中不含 $T$。
3. **A6 写成确定性形式**是因为 $(f,\kappa)$-robustness 本身是 worst-case 性质，且 tracker 侧不涉及投影。若 $\mathcal A_q$ 含内部随机性，把 (3.5) 换成 $\mathbb E\big[\|z_t-\bar q_t\|^2\mid\sigma(\{q_t^{(i)}\}_{i\in[N]})\big]\le\kappa_q V_{q,t}$，后文推导逐字成立。

### 3.3 Byzantine 数量与能力

**A8（Byzantine 数量）** 已知上界 $\hat B\ge B$，且

$$
N\ge2\hat B+3,
\qquad
\hat B<\frac N2 .
\tag{3.8}
$$

（第 I 部分仅通过 $\kappa_r,\kappa_q$ 使用该条件；§14 的定量常数在 $\hat B\le N/4$ 时为 $O(1)$。已知 Byzantine 数量上界是全部 Byzantine-robust 方法的共同前提。）

**A9（Byzantine 能力）** Byzantine 客户端可发送任意消息，唯一限制是 §4 的承诺机制；它们知晓 $\mathcal F_t$（含 honest 数据、模型与算法全部细节），但在 $V_t$、$I_t^{\mathrm{rnd}}$ 被抽取之前不知其取值。

### 3.4 本文**不**引入的假设

为明确分析质量的边界，这里显式声明本文**不**使用以下常见假设，以及原因：

| 常见假设 | 本文是否使用 | 说明 |
|---|---|---|
| 有界梯度 $\|\nabla f_i\|\le G$ 或有界随机梯度 | **完全不使用**（主定理与全部推论） | 见注 3.2。Adam 所需的 preconditioner 界由「只作用于二阶矩的 cap」**构造性地**保证，而非由假设提供。完整版中作为可选条件出现的 A10' 已删除 |
| 几乎必然有界 / 次高斯噪声 | **不使用** | Li–Rakhlin–Jadbabaie 2023、Hong–Lin 2023 去掉有界梯度所付的代价；本文只用 A3 的期望型有界方差 |
| 全局 Hessian 方差有界 | **不使用** | Byz-VR-MARINA / Byz-DASHA-PAGE / Byz-EF21 需要；本文与 RoSDHB、Byz-DM21 一样不需要 |
| 有界 Byzantine 消息范数 | **不使用** | A6 是 worst-case 性质，对任意 Byzantine 输入成立 |
| 周期性全梯度 / 有限和结构 | **不使用** | 纯随机梯度，适用于在线/流式设定 |
| 强凸性 | **不使用** | 非凸光滑设定 |

> **注 3.2（cap 的唯一职责，以及为什么不需要有界梯度）**
> 本版中 $C_{\max}$ **只作用于二阶矩累加器**（算法第 12 步），因此它只承担一件事：
>
> **Adam preconditioner 的确定性界——无需任何假设。**
> 由第 12 步的定义 $\hat g_t=\mathrm{clip}_{C_{\max}}(z_t)$ 有 $\|\hat g_t\|\le C_{\max}$，这是**算法的构造性保证**。于是
> $\tilde v_{t+1,j}\le(1-\beta_2)\sum_{\tau=0}^t\beta_2^{t-\tau}C_{\max}^2=(1-\beta_2^{t+1})C_{\max}^2$，
> bias correction 后**精确地** $v_{t+1,j}\le C_{\max}^2$，从而引理 11.1 的 $\lambda_A,\Lambda_A$ 成立。
> 通常 Adam 类非凸分析必须假设梯度或随机梯度有界才能界定 preconditioner
> （Reddi et al. 2021 假设逐坐标、几乎必然有界的随机梯度；Défossez et al. 2022 同；
> Li–Rakhlin–Jadbabaie 2023 与 Hong–Lin 2023 去掉有界梯度，但把噪声假设加强为几乎必然有界或次高斯，
> 且结论退化为高概率型；Gratton–Toint 2026 的统一框架亦声明其梯度预言机假设"necessarily stronger than bounded variance"）。
> **本算法把这一前提从"假设"变成了"算法性质"，且不付出任何裁剪偏差。**
>
> **为什么裁剪必须只作用于二阶矩。** 若像原版那样让 $\hat g_t$ 同时进入一阶矩，则更新方向含裁剪算子，
> 误差分解中出现残差 $\mathbb E(\|z_t\|-C_{\max})_+^2$。注 11.8a 证明：**仅有 A3（期望意义的有界方差）时，
> 该量的上确界等于二阶矩本身、与 $C_{\max}$ 无关**，因此无论怎样重排分析都不可能用问题参数把它闭合，
> 只能靠四阶矩或几乎必然有界性——这正是原版可选条件 A10' 的来源，也说明 A10' 不是"分析不够好"，而是原版结构下的必然。
> 变形 A 的做法不是更好地界住该项，而是让它**根本不出现**。

---

## 4. 承诺机制（唯一的协议前提）

> **Condition 1（single-round correction commitment / two-stage consistency）**
> 第 $t$ 轮每个客户端在 $V_t$ 的 seed 公布**之前**承诺唯一的底层 correction $r_t^{(i)}$；其后该轮所有上传均须由这同一个 $r_t^{(i)}$ 导出：
> $$
> P_t^{(i)}=\tfrac1{\sqrt r}R_t^{(i)}V_t,
> \qquad
> U_tc_t^{(i)}=Q_tr_t^{(i)} .
> \tag{4.1}
> $$

**这不是关于攻击者的假设，而是一条可强制执行的协议机制。** 实现方式：算法第 3 步之前各客户端先上传 $\mathrm{Commit}(r_t^{(i)})$（哈希承诺），服务器在收到 $P_t^{(i)},c_t^{(i)}$ 后校验一致性，不一致者本轮直接剔除（等价于减少有效 $B$，故不损害 A8）。Byzantine 仍可令 $r_t^{(i)}$ **任意恶意**——被限制的只是"事后按 $V_t$ 改写底层 correction"。

**理论作用。** Condition 1 使 $r_t^{(1)},\ldots,r_t^{(N)}$（**含 Byzantine 的**）成为 $\mathcal F_t^{c}$-可测量，从而与 $V_t$ 独立。这是第 II 部分全部 JL 论证的**唯一**前提。

**它是必要的（否则有具体攻击）。** 设无承诺机制，Byzantine 可在观察 $V_t$ 后再定 $r_t^{(i)}$。由于 $\mathrm{null}(V_t^\top)$ 的维数为 $n-r$（在 $r\ll n$ 的压缩区间内很大），攻击者可取

$$
R_t^{(i)}=\underbrace{R^{\mathrm{benign}}}_{\text{使 }P_t^{(i)}\text{ 看似正常}}+\ W,
\qquad WV_t=0,\ \ \|W\|_F\ \text{任意大},
$$

于是第一阶段的投影消息与诚实客户端无法区分（因此通过任何基于 $\{P_t^{(i)}\}$ 的 selector 检验），而第二阶段上传的真实行 $R_t^{(i)}[I_t,:]$ 却被 $W$ 任意污染。此时投影空间的鲁棒子集选择完全失效，A5 无论取多大 $\kappa_r$ 都不成立。故投影空间做鲁棒选择与承诺机制是绑定的。

> **注 4.1（原草稿的 Condition 2 与 Condition 3 是多余的，已删除）**
> 原草稿把"globally-consistent mask（$Q_t^{(1)}=\cdots=Q_t^{(N)}=Q_t$）"与"uniform random refresh（$I_t^{\mathrm{rnd}}$ 均匀无放回、$p>0$）"列为协议条件。二者**不是假设**：服务器在算法第 7 步生成唯一的 $I_t^{\mathrm{rnd}}$ 并广播唯一的 $I_t$，因此 mask 全局一致与 refresh 的均匀性都是**算法构造的事实**，$p=k_{\mathrm{rnd}}/(m-k_{\mathrm{top}})$ 是算法参数的函数，$p>0$ 由 (2.1) 保证。本文把它们并入 Algorithm 1 的描述，只在需要处以 (M2)、引理 6.1 的形式引用。
> 唯一真正需要单列的是 Condition 1，因为它约束的是**攻击者**的行为空间，需要密码学手段落实。

---

## 5. Filtration 与单轮随机性顺序

单轮内有五段随机性，必须区分。定义递增 $\sigma$-域

$$
\mathcal F_t
\subseteq\mathcal F_t^{g}
\subseteq\mathcal F_t^{c}
\subseteq\mathcal G_t
\subseteq\mathcal F_{t+1}.
\tag{5.1}
$$

| $\sigma$-域 | 已实现的信息 |
|---|---|
| $\mathcal F_t$ | 全部历史至第 $t-1$ 轮结束：$\theta_t$、$\{q_{t-1}^{(i)}\}_{i\in[N]}$、**$\{u_{t-1}^{(i)}\}_{i\in\mathcal H}$（客户端动量缓存）**、$m_t,\tilde v_t$ |
| $\mathcal F_t^{g}$ | $+\ \{g_t^{(i)}\}_{i\in\mathcal H}$（honest 随机梯度），从而 $\{u_t^{(i)}\}_{i\in\mathcal H}$（由 (2.1a) 是 $\mathcal F_t$ 与 $\{g_t^{(i)}\}$ 的确定性函数） |
| $\mathcal F_t^{c}$ | $+\ \{r_t^{(i)}\}_{i\in[N]}$，**含 Byzantine 的承诺 correction**（Condition 1） |
| $\mathcal G_t$ | $+\ V_t$，从而 $\{P_t^{(i)}\}$、$\mathcal S_t$、$\bar r_t^S$、$\Delta_t$、$\sigma_t$、$I_t^{\mathrm{top}}$、$Q_t^{\mathrm{top}}$、$\mathbf 1_{\mathcal E_t}$；但 $I_t^{\mathrm{rnd}}$ **尚未**抽取 |
| $\mathcal F_{t+1}$ | $+\ I_t^{\mathrm{rnd}}$，从而 $I_t,Q_t,\{q_t^{(i)}\},z_t,C_t,\hat g_t,\theta_{t+1}$ |

四条反复使用的可测性事实：

* **(M1)** $\bar g_t,\bar u_t,\bar r_t,V_{g,t},V_{u,t},V_{r,t},\{a_{t-1}^{(i)}\},V_{q,t-1}$ 均为 $\mathcal F_t^{c}$-可测（$\bar u_t,V_{u,t}$ 已是 $\mathcal F_t^{g}$-可测）；$\bar u_{t-1},h_{t-1},h_t$ 为 $\mathcal F_t$-可测（$\theta_t$ 在 $\mathcal F_t$ 中，见上表）——这是引理 11.4 中「交叉项期望为零」的依据；$V_t$ 独立于 $\mathcal F_t^{c}$（Condition 1）。
* **(M2)** $\bar r_t^S,\Delta_t,Q_t^{\mathrm{top}},\mathbf 1_{\mathcal E_t}$ 为 $\mathcal G_t$-可测；$I_t^{\mathrm{rnd}}$ 独立于 $\mathcal G_t$，且每个 $j\in\mathcal C_t$ 被选中的边缘概率恰为 $p$。
* **(M3)** $\{q_t^{(i)}\}_{i\in\mathcal H},V_{q,t},z_t,C_t,\hat g_t$ 为 $\mathcal F_{t+1}$-可测。
* **(M4)** A5 的 (3.4) 与 A7 的 (3.6) 都是**条件于 $\mathcal F_t^{c}$** 的陈述，故可与 (M1) 联合使用：对任意 $\mathcal F_t^{c}$-可测的 $X\ge0$，
 $$
 \mathbb E\big[X\,\mathbf 1_{\mathcal E_t^{\,c}}\mid\mathcal F_t^{c}\big]
 =X\cdot\Pr\big(\mathcal E_t^{\,c}\mid\mathcal F_t^{c}\big)
 \le\delta_{\mathrm{JL}}\,X .
 \tag{5.2}
 $$
 (5.2) 是 §9 保住无条件期望主定理的关键一步。

---
## 6. 压缩算子的两条 contraction

### 6.1 投影的基本性质

$Q_t,Q_t^{\mathrm{top}}$ 为行选择正交投影：

$$
Q_t^2=Q_t=Q_t^\top,\quad
Q_t(I-Q_t)=0,\quad
\|(I-Q_t)x\|\le\|x\|,
\tag{6.1}
$$

且由 $I_t^{\mathrm{top}}\subseteq I_t$，

$$
\|(I-Q_t)x\|^2\le\|(I-Q_t^{\mathrm{top}})x\|^2,
\qquad\forall x .
\tag{6.2}
$$

### 6.2 Uniform random refresh 的精确 contraction

> **引理 6.1** 对任意 $\mathcal G_t$-可测的随机向量 $x$，
> $$
> \mathbb E\big[\|(I-Q_t)x\|^2\mid\mathcal G_t\big]
> =(1-p)\,\big\|(I-Q_t^{\mathrm{top}})x\big\|^2
> \ \le\ (1-p)\|x\|^2 .
> \tag{6.3}
> $$

*证明.* $\|(I-Q_t)x\|^2=\sum_{j\notin I_t}\|X_{j,:}\|^2$。$j\in I_t^{\mathrm{top}}$ 必被选中；$j\in\mathcal C_t$ 独立于 $\mathcal G_t$ 地以概率 $\Pr(j\in I_t^{\mathrm{rnd}})=k_{\mathrm{rnd}}/(m-k_{\mathrm{top}})=p$ 被选中（均匀无放回抽样的边缘概率）。故

$$
\mathbb E\Big[\sum_{j\notin I_t}\|X_{j,:}\|^2\Big|\mathcal G_t\Big]
=\sum_{j\in\mathcal C_t}(1-p)\|X_{j,:}\|^2
=(1-p)\|(I-Q_t^{\mathrm{top}})x\|^2 .
$$

第二个不等号由 (6.1)。$\square$

> **注 6.2（该引理的力量）** (6.3) 对**任意历史 tracker 状态**成立，不要求 $x$ 与 Top-K 选择独立。这是本算法能在"有偏 Top-K + stateful tracker"下闭合 dispersion 的根本原因：Top-K 本身不提供对任意 $x$ 的 contraction（它只对被打分的那个向量提供），而 random refresh 提供的是**universal** contraction。

### 6.3 两条 contraction 的分工

* 引理 6.1（因子 $1-p$）：**无条件**成立，作用于**任意**向量，用于 §7 的 $V_{q,t}$ 与 §9 定理 9.2 的基础递推。
* A7（因子 $1-\omega_{\mathrm{top}}$）：仅在 $\mathcal E_t$ 上、仅作用于**被打分的** $\bar r_t^S$，用于 §9 定理 9.4 的精细递推。

(6.3) 的等式形式表明二者是**乘性**叠加的：$\mathbb E[\|(I-Q_t)x\|^2\mid\mathcal G_t]=(1-p)\|(I-Q_t^{\mathrm{top}})x\|^2$，故对 $x=\bar r_t$ 可先用 refresh 得因子 $(1-p)$，再在 $\mathcal E_t$ 上用 A7 处理 $(I-Q_t^{\mathrm{top}})$。§9 正是这样组织的。

### 6.4 两个初等不等式

**(Y)** $\|u+v\|^2\le(1+\nu)\|u\|^2+(1+\nu^{-1})\|v\|^2$，$\forall\nu>0$。
**注意**：本版中 $\beta$ 已被占用为**客户端动量参数**（(2.1a)），$\beta_1,\beta_2$ 为服务器 Adam 参数，故 (Y) 的自由参数一律记为 $\nu$。
**(J)** $\big\|\frac1k\sum_{i=1}^kx_i\big\|^2\le\frac1k\sum_{i=1}^k\|x_i\|^2$。

---

## 7. 客户端动量的性质与 Honest Dispersion

本节相对变形A 有实质改动：EF21 tracker 现在追踪的是**客户端动量缓存** $u_t^{(i)}$ 而非原始随机梯度 $g_t^{(i)}$。§7.1 先给出动量缓存的两条基本性质，§7.2–§7.3 是相应的 dispersion 界。

**记号约定。** 由 (2.1a)（$u_0^{(i)}=g_0^{(i)}$，$u_t^{(i)}=\beta u_{t-1}^{(i)}+(1-\beta)g_t^{(i)}$）展开，

$$
u_t^{(i)}=\sum_{k=0}^tw_{t,k}\,g_{t-k}^{(i)},
\qquad
w_{t,k}=\begin{cases}(1-\beta)\beta^k,&0\le k\le t-1,\\[2pt]\beta^t,&k=t,\end{cases}
\tag{7.0}
$$

这是一组**凸权重**：$w_{t,k}\ge0$ 且 $\sum_{k=0}^tw_{t,k}=(1-\beta^t)+\beta^t=1$。记

$$
\varrho_t:=\sum_{k=0}^tw_{t,k}^2
=(1-\beta)^2\frac{1-\beta^{2t}}{1-\beta^2}+\beta^{2t}
\ \le\ \frac{1-\beta}{1+\beta}+\beta^{2t} .
\tag{7.1}
$$

$\varrho_t$ 是**噪声抵消因子**：$\beta=0$ 时 $\varrho_t=1$（无动量），$\beta\to1$ 时 $\varrho_t\to\frac{1-\beta}{1+\beta}\to0$。它的时间平均满足

$$
\frac1T\sum_{t=0}^{T-1}\varrho_t\ \le\ \frac{1-\beta}{1+\beta}+\frac1{(1-\beta^2)T} .
\tag{7.2}
$$

### 7.1 动量缓存的 dispersion

> **引理 7.1（动量把噪声压小、把异质性原样保留）** 在 A2–A4 下，对一切 $t\ge0$，
> $$
> \sqrt{\mathbb E V_{u,t}}\ \le\ \zeta+\sigma\sqrt{\varrho_t}\ ,
> \tag{7.3}
> $$
> 从而
> $$
> \frac1T\sum_{t=0}^{T-1}\mathbb EV_{u,t}
> \ \le\ \Sigma_\beta^2+\frac{2\sigma^2}{(1-\beta^2)T},
> \qquad
> \boxed{\ \Sigma_\beta^2:=2\zeta^2+\frac{2(1-\beta)}{1+\beta}\sigma^2\ }.
> \tag{7.4}
> $$

*证明.* 记 $\delta^{(i)}(\theta):=\nabla f_i(\theta)-\nabla f(\theta)$、$\xi_t^{(i)}:=g_t^{(i)}-\nabla f_i(\theta_t)$、$\bar\xi_t:=\frac1H\sum_{i\in\mathcal H}\xi_t^{(i)}$。由 (7.0)，

$$
u_t^{(i)}-\bar u_t
=\underbrace{\sum_kw_{t,k}\,\delta^{(i)}(\theta_{t-k})}_{\text{信号部分}}
+\underbrace{\sum_kw_{t,k}\big(\xi_{t-k}^{(i)}-\bar\xi_{t-k}\big)}_{\text{噪声部分}} .
$$

在乘积测度 $\mathbb P\otimes\mathrm{Unif}(\mathcal H)$ 上视二者为 $L^2$ 元素，由 Minkowski 不等式，$\sqrt{\mathbb EV_{u,t}}$ 不超过两部分 $L^2$ 范数之和。

**信号部分.** 权重为凸组合，由 Jensen 与 A4，

$$
\frac1H\sum_{i\in\mathcal H}\Big\|\sum_kw_{t,k}\delta^{(i)}(\theta_{t-k})\Big\|^2
\le\sum_kw_{t,k}\cdot\frac1H\sum_{i\in\mathcal H}\big\|\delta^{(i)}(\theta_{t-k})\big\|^2
\le\sum_kw_{t,k}\,\zeta^2=\zeta^2 .
$$

**噪声部分.** 不同轮次的采样是 fresh 的，故 $\{\xi_\tau^{(i)}-\bar\xi_\tau\}_\tau$ 关于 $\{\mathcal F_\tau\}$ 是鞅差列，且 $w_{t,k}$ 为确定性权重，交叉项期望为零：

$$
\mathbb E\Big\|\sum_kw_{t,k}\big(\xi_{t-k}^{(i)}-\bar\xi_{t-k}\big)\Big\|^2
=\sum_kw_{t,k}^2\,\mathbb E\big\|\xi_{t-k}^{(i)}-\bar\xi_{t-k}\big\|^2 .
$$

对 $i\in\mathcal H$ 取平均，由 $\frac1H\sum_i\|\xi^{(i)}-\bar\xi\|^2\le\frac1H\sum_i\|\xi^{(i)}\|^2\le\sigma^2$（A3）得该部分 $\le\varrho_t\sigma^2$。

合并即 (7.3)。(7.4) 由 $(\zeta+\sigma\sqrt{\varrho_t})^2\le2\zeta^2+2\varrho_t\sigma^2$ 与 (7.2)。$\square$

> **注 7.2（这一条是本版的全部动机）** (7.4) 说明动量对 dispersion 的两个成分作用完全不同：
> $$
> \zeta^2\ \text{（异质性）：不变}\qquad\qquad
> \sigma^2\ \text{（采样噪声）：乘上 }\tfrac{1-\beta}{1+\beta} .
> $$
> 这正是我们想要的分工——异质性项有匹配下界（$\Omega(\frac BN\zeta^2)$，见 §15.2），本就不可能消除；而噪声项是变形A 中 error floor 不随步长消失的病灶。$\beta=0.9$ 时噪声被压到约 $1/19$，$\beta=0.99$ 时约 $1/199$。
> 这一机制与 Karimireddy–He–Jaggi (ICML 2021) 引入客户端动量以增强 Byzantine 鲁棒性的论证是同一个：鲁棒聚合器的误差正比于诚实方 dispersion，压小 dispersion 就直接压小 error floor。

> **注 7.3（$\varrho_t$ 的暂态项）** (7.1) 中的 $\beta^{2t}$ 反映动量缓存的预热：$t=0$ 时 $u_0=g_0$，没有任何噪声抵消。由于它对时间平均只贡献 $O(1/T)$（(7.2)），后文一律并入 $O(1/T)$ 项。这也是本版把 §7 的结论全部写成**时间平均**形式（而非变形A 的逐 $t$ 上确界形式）的原因；(11.x) 之后所有用到 dispersion 的地方本来就只需要时间平均。

### 7.2 Tracker dispersion

> **定理 7.4** 在 A2–A4 下，
> $$
> \mathbb EV_{q,t}\ \le\ (1-p)\,\mathbb EV_{q,t-1}+\mathbb EV_{u,t},
> \tag{7.5}
> $$
> 从而
> $$
> \bar V_q^{(T)}:=\frac1T\sum_{t=0}^{T-1}\mathbb EV_{q,t}
> \ \le\ \frac1p\cdot\frac1T\sum_{t=0}^{T-1}\mathbb EV_{u,t}
> \ \le\ \frac{\Sigma_\beta^2}p+\frac{2\sigma^2}{p(1-\beta^2)T} .
> \tag{7.6}
> $$

*证明.* 记 $a_t^{(i)}=q_t^{(i)}-\bar q_t$、$b_t^{(i)}=u_t^{(i)}-\bar u_t$（$i\in\mathcal H$）。因全体 honest 客户端使用同一 $Q_t$（算法第 7 步），对 (2.6) 取 honest 平均得 $\bar q_t=(I-Q_t)\bar q_{t-1}+Q_t\bar u_t$，两式相减：

$$
a_t^{(i)}=(I-Q_t)a_{t-1}^{(i)}+Q_tb_t^{(i)} .
\tag{7.7}
$$

由 (6.1) 有 $\langle(I-Q_t)u,Q_tv\rangle=0$，故正交分解 $\|a_t^{(i)}\|^2=\|(I-Q_t)a_{t-1}^{(i)}\|^2+\|Q_tb_t^{(i)}\|^2$。对 $i\in\mathcal H$ 取平均：第二项逐点 $\le\|b_t^{(i)}\|^2$，故其平均 $\le V_{u,t}$；第一项中 $\{a_{t-1}^{(i)}\}$ 为 $\mathcal F_t\subseteq\mathcal G_t$-可测，由引理 6.1 与 (6.2) 得条件期望 $\le(1-p)V_{q,t-1}$。取全期望即 (7.5)。

对 (7.5) 从 $t=0$ 到 $T-1$ 求和，用 $V_{q,-1}=0$：

$$
\sum_{t=0}^{T-1}\mathbb EV_{q,t}\le(1-p)\sum_{t=0}^{T-1}\mathbb EV_{q,t}+\sum_{t=0}^{T-1}\mathbb EV_{u,t}
\ \Longrightarrow\
p\sum_{t=0}^{T-1}\mathbb EV_{q,t}\le\sum_{t=0}^{T-1}\mathbb EV_{u,t},
$$

除以 $pT$ 并代入 (7.4) 即得 (7.6)。$\square$

> **注 7.5（改为时间平均是无损的）** 变形A 的定理 7.1 给出的是逐 $t$ 的上确界 $\mathbb EV_{q,t}\le\Sigma^2/p$。本版改用时间平均 (7.6)，常数完全相同（几何级数求和与直接对递推求和给出同一个 $1/p$），但避免了处理 $\varrho_t$ 暂态所需的逐 $t$ 上确界。后文所有用到 $V_{q,t}$ 的地方（推论 8.2 经 $\Delta_{\mathrm{est},t}$ 进入主定理）都只需要时间平均，故无损失。
> **但 $V_{r,t}$ 是例外**：它经 (9.11) 进入**逐 $t$** 的 tracking-error 递推，而变形A 的引理 9.7 要求那里的加性项是常数。本版为此把引理 9.7 放宽为允许 $\Xi_t$ 依赖 $t$（注 9.7a），放宽后常数完全不变。

### 7.3 Correction dispersion

> **定理 7.6** 在同样条件下，
> $$
> \bar V_r^{(T)}:=\frac1T\sum_{t=0}^{T-1}\mathbb EV_{r,t}
> \ \le\ \Big(1+\tfrac1{\sqrt p}\Big)^2
> \bigg[\Sigma_\beta^2+\frac{2\sigma^2}{(1-\beta^2)T}\bigg] .
> \tag{7.8}
> $$

*证明.* 由 $r_t^{(i)}=u_t^{(i)}-q_{t-1}^{(i)}$ 得 $r_t^{(i)}-\bar r_t=b_t^{(i)}-a_{t-1}^{(i)}$，故 $V_{r,t}=\frac1H\sum_{i\in\mathcal H}\|b_t^{(i)}-a_{t-1}^{(i)}\|^2$。记 $\mathcal U:=\frac1T\sum_{t=0}^{T-1}\mathbb EV_{u,t}$。在乘积测度 $\mathbb P\otimes\mathrm{Unif}(\mathcal H)\otimes\mathrm{Unif}(\{0,\ldots,T-1\})$ 上用 Minkowski：

$$
\sqrt{\bar V_r^{(T)}}\ \le\ \sqrt{\mathcal U}+\sqrt{\tfrac1T\sum_{t=0}^{T-1}\mathbb EV_{q,t-1}} .
$$

**关键：不要把余项拆成加性的 $O(T^{-1/2})$ 再平方**（那样交叉项是 $O(T^{-1/2})$ 而非 $O(1/T)$，在 $\beta$ 随 $T$ 变化时更会恶化到 $O(T^{-1/4})$，见注 7.6a）。正确做法是**整体提出公因子**：由 $V_{q,-1}=0$ 得 $\frac1T\sum_{t=0}^{T-1}\mathbb EV_{q,t-1}\le\bar V_q^{(T)}$，再由 (7.6) 的**第一个**不等号 $\bar V_q^{(T)}\le\mathcal U/p$，

$$
\sqrt{\bar V_r^{(T)}}\ \le\ \sqrt{\mathcal U}+\frac{\sqrt{\mathcal U}}{\sqrt p}
=\Big(1+\tfrac1{\sqrt p}\Big)\sqrt{\mathcal U}
\quad\Longrightarrow\quad
\bar V_r^{(T)}\le\Big(1+\tfrac1{\sqrt p}\Big)^2\mathcal U ,
$$

最后代入 (7.4) 的 $\mathcal U\le\Sigma_\beta^2+\frac{2\sigma^2}{(1-\beta^2)T}$。$\square$

> **注 7.6a（为什么这一步必须提公因子）** 若按"$\sqrt{\bar V_r}\le\Sigma_\beta(1+p^{-1/2})+\varepsilon_T$ 然后平方"的写法，交叉项 $2\Sigma_\beta(1+p^{-1/2})\varepsilon_T$ 的阶是 $\varepsilon_T=O\big((1-\beta^2)^{-1/2}T^{-1/2}\big)$。在推论 12.4 的取参 $1-\beta=\Theta(T^{-1/2})$ 下这是 $\Theta(T^{-1/4})$，它经 $\Xi_\chi\to\Phi_\chi$ 进入主定理，会把**定理 B 分支**的速率打成 $O(T^{-1/4})$、复杂度打成 $O(\varepsilon^{-8})$。提公因子后余项是 $\big(1+p^{-1/2}\big)^2\frac{2\sigma^2}{(1-\beta^2)T}$，与 $\mathcal O_1$ 中其他 $\frac1{(1-\beta)T}$ 型项同类，取参后为 $\Theta(T^{-1/2})$，与主项同阶，速率得以保住。

> **注 7.7（这一节为什么仍是整套分析的关键）**
> 与变形A 相同：在"有偏压缩 + stateful tracker + Byzantine"三者叠加时，最难的一步是证明 honest dispersion **不随 $T$ 发散**；A5、A6 把 Byzantine 造成的偏差全部归结到 $V_{r,t}$、$V_{q,t}$ 上，若这两者可能增长，则鲁棒性保证形同虚设。纯 Top-K 给不出 (7.5)——它只对被打分的那**一个**向量提供 contraction，而 $a_{t-1}^{(i)}$ 是 $H$ 个与打分无关的向量。**uniform random refresh 的 universal contraction（引理 6.1）正是为此设计的。**
> 本版新增的信息是：把 $g$ 换成 $u$ 之后，(7.5) 的结构**一字未变**（动量只改变输入的 dispersion，不改变 tracker 的递推结构），而输入的 dispersion 从 $\Sigma^2$ 降到 $\Sigma_\beta^2$。

---

## 8. 两个鲁棒模块的误差

直接组合 A5、A6 与 §7：

> **推论 8.1（correction 侧）** 在 A5、A7 下，逐 $t$ 有 $\mathbb E[\|\Delta_t\|^2\mathbf 1_{\mathcal E_t}]\le\kappa_r\mathbb EV_{r,t}$，从而由定理 7.6，
> $$
> \frac1T\sum_{t=0}^{T-1}\mathbb E\Big[\big\|\Delta_t\big\|^2\mathbf 1_{\mathcal E_t}\Big]
> \ \le\ \kappa_r\,\bar V_r^{(T)}
> \ \le\ \kappa_r\Big(1+\tfrac1{\sqrt p}\Big)^2\Sigma_\beta^2+O(1/T) .
> \tag{8.1}
> $$

> **推论 8.2（tracker 侧）** 在 A6 下（本版中 $\mathcal A_q=\mathrm{MK}_{s_q}\circ\mathrm{ARC}$，鲁棒常数记为 $\kappa_q^\star$，见推论 10.7），逐 $t$ 有 $\mathbb E\|z_t-\bar q_t\|^2\le\kappa_q^\star\mathbb EV_{q,t}$，从而由定理 7.4，
> $$
> \frac1T\sum_{t=0}^{T-1}\mathbb E\big\|z_t-\bar q_t\big\|^2
> \ \le\ \kappa_q^\star\,\bar V_q^{(T)}
> \ \le\ \frac{\kappa_q^\star}p\Sigma_\beta^2+O(1/T),
> \qquad
> \Sigma_\beta^2=2\zeta^2+\frac{2(1-\beta)}{1+\beta}\sigma^2 .
> \tag{8.2}
> $$

(8.2) 说明：**Byzantine 客户端在 stateful tracker 中累积的历史恶意状态，对服务器最终方向的影响被控制在 $O(\kappa_q^\star\Sigma_\beta^2/p)$**。这一项将直接构成主定理的 error floor 主项。相对变形A 的两点变化：
* **$\kappa_q^\star$ 已由推论 10.7 收紧到 $\le73$**（$B=\hat B$ 时 $\le41$），ARC 预裁剪的全部理论代价就是 $\kappa_q$ 上一个不超过 $1$ 的加性项；
* **$\Sigma^2$ 已被 $\Sigma_\beta^2$ 取代**：其中的采样噪声部分带有因子 $\frac{1-\beta}{1+\beta}$，$\beta\to1$ 时该项消失，只剩 $2\zeta^2$——而 $\zeta^2$ 部分有匹配下界（§15.2），本就不可消除。

---

## 9. 平均 EF21 Tracking Error：两条互补递推

这是本文最核心的技术部分，也是原草稿"问题 §1"（主定理未用到 $\kappa_r,\omega_{\mathrm{top}}$）与"问题 §3"（JL good event 的概率形式）的解决处。

### 9.1 tracking error 的递推结构

> **引理 9.1** 对一切 $t\ge0$，
> $$
> e_t=-(I-Q_t)\,\bar r_t ,
> \tag{9.1}
> $$
> 且对 $t\ge1$，
> $$
> \bar r_t=-e_{t-1}+(\bar u_t-\bar u_{t-1}),
> \qquad
> \bar r_0=\bar u_0 .
> \tag{9.2}
> $$

*证明.* 对 (2.6) 取 honest 平均：$\bar q_t=\bar q_{t-1}+Q_t(\bar u_t-\bar q_{t-1})=\bar q_{t-1}+Q_t\bar r_t$。于是

$$
e_t=\bar q_t-\bar u_t=\bar q_{t-1}+Q_t\bar r_t-\bar u_t=-\bar r_t+Q_t\bar r_t=-(I-Q_t)\bar r_t .
$$

又 $\bar r_t=\bar u_t-\bar q_{t-1}=\bar u_t-\bar u_{t-1}-(\bar q_{t-1}-\bar u_{t-1})=-e_{t-1}+(\bar u_t-\bar u_{t-1})$；$t=0$ 时 $\bar q_{-1}=0$。$\square$

### 9.2 递推 I：仅用 random refresh（无条件，不需 A5/A7）

> **定理 9.2** 在 A2–A4 下，对 $t\ge1$，
> $$
> E_t\ \le\ \Big(1-\frac p2\Big)E_{t-1}+\frac2p\,D_t ,
> \tag{9.3}
> $$
> 且 $E_0\le(1-p)\,\mathbb E\|\bar u_0\|^2\le\|\nabla f(\theta_0)\|^2+\dfrac{\sigma^2}H$。

*证明.* 由 (9.1) 与引理 6.1（取 $x=\bar r_t$，它是 $\mathcal F_t^{c}\subseteq\mathcal G_t$-可测的），

$$
\mathbb E\big[\|e_t\|^2\mid\mathcal G_t\big]
=(1-p)\big\|(I-Q_t^{\mathrm{top}})\bar r_t\big\|^2
\le(1-p)\|\bar r_t\|^2 .
\tag{9.4}
$$

代入 (9.2) 并对 $\nu>0$ 用 (Y)：

$$
E_t\le(1-p)\Big[(1+\beta)E_{t-1}+(1+\beta^{-1})D_t\Big].
$$

取 $\nu=p/2$：$(1-p)(1+\frac p2)=1-\frac p2-\frac{p^2}2\le1-\frac p2$，而 $(1-p)(1+\frac2p)=\frac{2-p-p^2}p\le\frac2p$。得 (9.3)。$t=0$ 时由 (9.1)、(9.4) 与 $\bar r_0=\bar u_0$ 得 $E_0\le(1-p)\mathbb E\|\bar u_0\|^2$，再由 $\mathbb E\|\bar u_0\|^2=\|h_0\|^2+\mathbb E\|\bar u_0-h_0\|^2\le\|h_0\|^2+\frac{\sigma^2}H$（(3.1)）。$\square$

### 9.3 关键引理：good event 的无条件处理

原草稿认为，由于 A5、A7 只在 $\mathcal E_t$ 上成立，而 $\mathcal E_t$ 之外 Byzantine correction 可任意大，主定理必须写成 high-probability 形式，或需额外引入第一阶段的范数控制。下面说明**二者都不必要**。

> **引理 9.3（$\mathcal E_t$ 之外不出现 Byzantine 量）** 在 A5、A7 下，若
> $$
> \delta_{\mathrm{JL}}\le\frac{\omega_{\mathrm{top}}}4 ,
> \tag{9.5}
> $$
> 则对一切 $t\ge0$，
> $$
> \mathbb E\Big[\big\|(I-Q_t^{\mathrm{top}})\bar r_t\big\|^2\ \Big|\ \mathcal F_t^{c}\Big]
> \ \le\
> \Big(1-\frac{\omega_{\mathrm{top}}}4\Big)\|\bar r_t\|^2
> +\frac{10}{\omega_{\mathrm{top}}}\,\kappa_r\,V_{r,t}
> \qquad\text{a.s.}
> \tag{9.6}
> $$

*证明.* 记 $\omega=\omega_{\mathrm{top}}$。把 $\mathcal E_t$ 与其补分开处理。

**(i) 在 $\mathcal E_t$ 上。** 由 $\bar r_t=\bar r_t^S-\Delta_t$ 与 (Y)（参数 $\nu$），再用 $\|(I-Q_t^{\mathrm{top}})\Delta_t\|\le\|\Delta_t\|$ 与 A7 的 (3.7)：

$$
\big\|(I-Q_t^{\mathrm{top}})\bar r_t\big\|^2
\le(1+\nu)(1-\omega)\big\|\bar r_t^S\big\|^2+(1+\nu^{-1})\|\Delta_t\|^2 .
$$

又由 $\bar r_t^S=\bar r_t+\Delta_t$ 与 (Y)（参数 $\nu'$）：$\|\bar r_t^S\|^2\le(1+\nu')\|\bar r_t\|^2+(1+\nu'^{-1})\|\Delta_t\|^2$。取 $\nu=\nu'=\omega/4$。由

$$
\Big(1+\frac\omega4\Big)^2(1-\omega)
=1-\frac\omega2-\frac{7\omega^2}{16}-\frac{\omega^3}{16}
\le1-\frac\omega2,
$$

以及 $\big(1+\frac\omega4\big)(1-\omega)=1-\frac{3\omega}4-\frac{\omega^2}4\le1$ 与 $\omega\le1$，$\Delta_t$ 的总系数

$$
\Big(1+\frac\omega4\Big)(1-\omega)\Big(1+\frac4\omega\Big)+\Big(1+\frac4\omega\Big)
\le2\Big(1+\frac4\omega\Big)\le\frac2\omega+\frac8\omega=\frac{10}\omega .
$$

故在 $\mathcal E_t$ 上

$$
\big\|(I-Q_t^{\mathrm{top}})\bar r_t\big\|^2
\le\Big(1-\frac\omega2\Big)\|\bar r_t\|^2+\frac{10}\omega\|\Delta_t\|^2 .
\tag{9.7}
$$

**(ii) 在 $\mathcal E_t^{\,c}$ 上。** 此处 A5、A7 均不可用，但 $(I-Q_t^{\mathrm{top}})$ 是正交投影，故**无条件地**

$$
\big\|(I-Q_t^{\mathrm{top}})\bar r_t\big\|^2\le\|\bar r_t\|^2 .
\tag{9.8}
$$

**关键之处在于 (9.8) 的右端只含 $\bar r_t$——一个纯 honest 量**（$\bar r_t=\frac1H\sum_{i\in\mathcal H}r_t^{(i)}$ 不含任何 Byzantine 分量），且由 (M1) 它是 $\mathcal F_t^{c}$-可测的。因此由 (5.2)，

$$
\mathbb E\Big[\big\|(I-Q_t^{\mathrm{top}})\bar r_t\big\|^2\mathbf 1_{\mathcal E_t^{\,c}}\Big|\mathcal F_t^{c}\Big]
\le\|\bar r_t\|^2\Pr\big(\mathcal E_t^{\,c}\mid\mathcal F_t^{c}\big)
\le\delta_{\mathrm{JL}}\|\bar r_t\|^2 .
\tag{9.9}
$$

**(iii) 合并。** 由 (9.7)、A5 的 (3.4) 与 (9.9)：

$$
\mathbb E\Big[\big\|(I-Q_t^{\mathrm{top}})\bar r_t\big\|^2\Big|\mathcal F_t^{c}\Big]
\le\Big(1-\frac\omega2+\delta_{\mathrm{JL}}\Big)\|\bar r_t\|^2
+\frac{10}\omega\kappa_rV_{r,t},
$$

代入 (9.5) 即得 (9.6)。$\square$

> **注 9.4（这条引理消除了三个技术障碍）**
> 1. **不需要 high-probability 定理。** (9.6) 是无条件的条件期望不等式，可直接嵌入期望型 descent 分析，主定理仍是标准的 $\frac1T\sum\mathbb E\|\nabla f\|^2$ 形式。
> 2. **不需要对 $T$ 做 union bound。** (9.5) 只要求**每轮**的失败概率 $\le\omega_{\mathrm{top}}/4$，与 $T$ 无关；因此实例化时 $\log\frac1{\delta_{\mathrm{JL}}}$ 中不含 $T$（对比原草稿建议的 $\delta_t=\delta/T$ 与 $r\gtrsim\varepsilon^{-2}\log\frac{mT\binom Ns}\delta$）。
> 3. **不需要重新引入有界 Byzantine 消息假设。** 失败事件上出现的唯一量是 honest 均值 $\bar r_t$。
>
> 其代价仅是把失败概率从"任意小"收紧到"$\le\omega_{\mathrm{top}}/4$"——这是一个 $O(1)$ 量级的常数要求，对投影维数只贡献 $\log\frac1{\omega_{\mathrm{top}}}$。

### 9.4 递推 II：refresh 与 Top-K 的乘性叠加

> **定理 9.5** 在 A2–A7 与 (9.5) 下，定义**联合 contraction 系数**
> $$
> \chi\ :=\ 1-(1-p)\Big(1-\frac{\omega_{\mathrm{top}}}4\Big)
> \ \ \ge\ \max\Big\{p,\ \frac{\omega_{\mathrm{top}}}4\Big\},
> \tag{9.10}
> $$
> 则对 $t\ge1$，
> $$
> E_t\ \le\ \Big(1-\frac\chi2\Big)E_{t-1}
> +\frac2\chi D_t
> +\frac{10\,\kappa_r}{\omega_{\mathrm{top}}}\,\mathbb E V_{r,t} .
> \tag{9.11}
> $$

*证明.* 由 (9.1) 与引理 6.1（$x=\bar r_t$ 为 $\mathcal G_t$-可测），再对 $\mathcal F_t^{c}$ 取条件期望并用塔性质与引理 9.3：

$$
\mathbb E\big[\|e_t\|^2\mid\mathcal F_t^{c}\big]
=(1-p)\,\mathbb E\Big[\big\|(I-Q_t^{\mathrm{top}})\bar r_t\big\|^2\Big|\mathcal F_t^{c}\Big]
\le(1-p)\Big(1-\frac{\omega_{\mathrm{top}}}4\Big)\|\bar r_t\|^2
+\frac{10\kappa_r}{\omega_{\mathrm{top}}}V_{r,t},
$$

其中末项用了 $1-p\le1$。按 (9.10)，前一系数即 $1-\chi$。取全期望，代入 (9.2) 并用 (Y)（参数 $\nu=\chi/2$）：

$$
E_t\le(1-\chi)\Big[\Big(1+\frac\chi2\Big)E_{t-1}+\Big(1+\frac2\chi\Big)D_t\Big]
+\frac{10\kappa_r}{\omega_{\mathrm{top}}}\mathbb EV_{r,t},
$$

再用 $(1-\chi)(1+\frac\chi2)\le1-\frac\chi2$ 与 $(1-\chi)(1+\frac2\chi)\le\frac2\chi$ 即得 (9.11)。$\square$

> **注 9.6（两个模块的分工在此显式化，且 trade-off 是真实的）**
> (9.10) 表明 refresh 与 Top-K 的 contraction **乘性叠加**：$1-\chi=(1-p)(1-\frac{\omega_{\mathrm{top}}}4)$。这来自引理 6.1 的**等式**形式
> $\mathbb E[\|(I-Q_t)x\|^2\mid\mathcal G_t]=(1-p)\|(I-Q_t^{\mathrm{top}})x\|^2$——先由 refresh 得因子 $(1-p)$，再在 $\mathcal E_t$ 上由 Top-K 处理 $(I-Q_t^{\mathrm{top}})$。
> 于是原草稿期望的两条互补作用得到严格实现：
> $$
> p\ \longrightarrow\ \text{individual tracker stability （定理 7.4、7.6）},
> \qquad
> \omega_{\mathrm{top}}\ \longrightarrow\ \text{average tracking efficiency （定理 9.5）}.
> $$
> 但 (9.3) 与 (9.11) **不是简单的强弱关系**：(9.11) 的 contraction 更强（$\chi\ge p$），代价是多出常数项 $\frac{10\kappa_r}{\omega_{\mathrm{top}}}\bar V_r$；而 (9.3) 完全无 $\kappa_r$ 项，且不依赖 A5/A7（故不依赖任何投影分析）。主定理因此分层为定理 A（用 (9.3)）与定理 B（用 (9.11)），最终界取二者较小者。这个 trade-off 由 $\rho$ 调节，§13.3 给出 $\rho^\star$。

### 9.5 时间平均

> **引理 9.7（$\Xi$ 允许依赖 $t$）** 设 $\{E_t\}$ 非负有限（引理 11.0'）且满足
> $$
> E_t\le\Big(1-\frac c2\Big)E_{t-1}+\frac2cD_t+\Xi_t
> \qquad(t\ge1,\ c\in(0,1],\ \Xi_t\ge0),
> $$
> 则
> $$
> \frac1T\sum_{t=0}^{T-1}E_t
> \ \le\
> \frac{2E_0}{cT}
> +\frac4{c^2}\cdot\frac1T\sum_{t=1}^{T-1}D_t
> +\frac2c\cdot\frac1T\sum_{t=1}^{T-1}\Xi_t .
> \tag{9.12}
> $$

*证明.* 对 $t=1,\ldots,T-1$ 求和：$\sum_{t=1}^{T-1}E_t\le(1-\frac c2)\sum_{t=0}^{T-2}E_t+\frac2c\sum_{t=1}^{T-1}D_t+\sum_{t=1}^{T-1}\Xi_t$。两边加 $E_0$ 并注意 $\sum_{t=0}^{T-2}E_t\le\sum_{t=0}^{T-1}E_t$：

$$
\sum_{t=0}^{T-1}E_t\le E_0+\Big(1-\frac c2\Big)\sum_{t=0}^{T-1}E_t+\frac2c\sum_{t=1}^{T-1}D_t+\sum_{t=1}^{T-1}\Xi_t .
$$

由引理 11.0'，$\sum_{t=0}^{T-1}E_t<\infty$，故可移项得 $\frac c2\sum_{t=0}^{T-1}E_t\le E_0+\frac2c\sum_{t=1}^{T-1}D_t+\sum_{t=1}^{T-1}\Xi_t$，除以 $\frac{cT}2$ 即得。$\square$

> **注 9.7a（为什么必须允许 $\Xi_t$ 依赖 $t$）** 定理 9.5 的 (9.11) 末项是 $\frac{10\kappa_r}{\omega_{\mathrm{top}}}\mathbb EV_{r,t}$，**逐 $t$ 变化**。在变形A 中这不成问题，因为那里的定理 7.2 给出逐 $t$ 的常数界 $\mathbb EV_{r,t}\le(1+p^{-1/2})^2\Sigma^2$；本版的定理 7.6 只给**时间平均**（§7 改写的代价），故必须把引理 9.7 的 $\Xi$ 放宽为 $\Xi_t$。放宽后 $\frac2c\cdot\frac1T\sum_{t=1}^{T-1}\Xi_t\le\frac2c\cdot\frac{10\kappa_r}{\omega_{\mathrm{top}}}\bar V_r^{(T)}=\frac{20\kappa_r\bar V_r^{(T)}}{c\,\omega_{\mathrm{top}}}$（各项非负，$\sum_{t=1}^{T-1}\le\sum_{t=0}^{T-1}$），与 (9.14) 的 $\Xi_\chi$ 完全吻合，**常数不变**。

> **注 9.8** 引理 9.7 显式区分了 $E_0$ 与 $\sum_{t\ge1}D_t$：递推 (9.3)、(9.11) 只对 $t\ge1$ 成立（因 (9.2) 需要 $\bar u_{t-1}$），$D_0$ 未定义。原草稿的式 (49)–(50) 未区分这一点。$E_0$ 已由定理 9.2 界定，与 $T$ 无关，故只贡献 $O(1/T)$。

综合定理 9.2、9.5 与引理 9.7，并记 $\bar D^{(T)}:=\frac1T\sum_{t=1}^{T-1}D_t$：

$$
\bar E^{(T)}:=\frac1T\sum_{t=0}^{T-1}E_t
\ \le\
\min\Big\{\mathcal T_A,\ \mathcal T_B\Big\},
\tag{9.13}
$$

$$
\mathcal T_A=\frac{2E_0}{pT}+\frac4{p^2}\bar D^{(T)},
\qquad
\mathcal T_B=\frac{2E_0}{\chi T}+\frac4{\chi^2}\bar D^{(T)}
+\frac{20\,\kappa_r}{\chi\,\omega_{\mathrm{top}}}\bar V_r^{(T)},
\qquad
\bar V_r^{(T)}\le\Big(1+\frac1{\sqrt p}\Big)^2\Big[\Sigma_\beta^2+\frac{2\sigma^2}{(1-\beta^2)T}\Big]\ \text{（定理 7.6）}.
\tag{9.14}
$$

---

## 10. ARC 预聚合裁剪：鲁棒常数与输出有界性

本节替换原版的「Clipping Residual 的显式闭合」。**原版第 10 节整节（order-statistic 引理 10.1、注 10.2、命题 10.3、定理 10.4、推论 10.6、可选条件 A10'）全部删除**：在新算法中，裁剪不再是聚合器之外的独立模块，而是被吸收进 $(f,\kappa)$-robustness，因此**不再产生任何残差项**。

### 10.1 ARC 的定义与阈值性质

> **定义 10.1（Adaptive Robust Clipping, ARC；Allouah et al. 2025）**
> 输入 $x_1,\ldots,x_N\in\mathbb R^D$ 与容忍上界 $\hat B$。令
> $$
> k_{\mathrm{arc}}:=\Big\lfloor 2\tfrac{\hat B}N\big(N-\hat B\big)\Big\rfloor ,
> \tag{10.1}
> $$
> 按范数降序排列得置换 $\pi$（$\|x_{\pi(1)}\|\ge\cdots\ge\|x_{\pi(N)}\|$），取阈值
> $$
> C^{\mathrm{arc}}:=\big\|x_{\pi(k_{\mathrm{arc}}+1)}\big\| ,
> \tag{10.2}
> $$
> 输出 $\mathrm{ARC}(x_1,\ldots,x_N):=\big(\mathrm{clip}_{C^{\mathrm{arc}}}(x_1),\ldots,\mathrm{clip}_{C^{\mathrm{arc}}}(x_N)\big)$，其中 $\mathrm{clip}_C(x)=x\min\{1,C/\|x\|\}$。

> **引理 10.2（阈值的两条性质）** 设 $N>2\hat B$。则
> **(a)** $k_{\mathrm{arc}}+1\ge\hat B+1$，即至少 $\hat B+1$ 个输入的范数不小于 $C^{\mathrm{arc}}$；
> **(b)** 对任意 $S\subseteq[N]$ 且 $|S|=N-\hat B$，有 $\max_{i\in S}\|x_i\|\ge C^{\mathrm{arc}}$。

*证明.* (a) 由 $2\tfrac{\hat B}N(N-\hat B)-\hat B=\hat B\big(1-\tfrac{2\hat B}N\big)>0$（$\hat B<N/2$）得 $k_{\mathrm{arc}}+1>\hat B$，因 $k_{\mathrm{arc}}+1$ 为整数故 $\ge\hat B+1$。
(b) 由 (a)，范数小于 $C^{\mathrm{arc}}$ 的输入至多 $N-\hat B-1$ 个，故任意大小 $N-\hat B$ 的子集必含至少一个范数 $\ge C^{\mathrm{arc}}$ 的输入。$\square$

> **注 10.3（$k_{\mathrm{arc}}$ 不是自由参数）** 原版 §2.1 允许 $k_{\mathrm{clip}}$ 在 $[\hat B,H-2]$ 内任取，并在注 2.1 中批评原草稿的取法「$\alpha$ 未定义、算法无从计算」。这一批评基于误读：(10.1) 中的比例是**容忍参数** $\hat B/N$（算法已知），不是真实的 $B/N$。而且引理 10.2 表明这个特定取法是必需的——它同时保证阈值不塌陷（(a)）与输出有界性（(b)），二者都是下面两条定理的前提。因此本版把 $k_{\mathrm{clip}}$ 改回 (10.1) 的取法，原版条件 (2.0) 作废，只保留 A8 的 $N\ge2\hat B+3$。

### 10.2 鲁棒性保持

> **定理 10.4（ARC 保持 $(f,\kappa)$-robustness；Allouah et al. 2025, Thm 3.2）**
> 设 $\mathcal A$ 为 $(\hat B,\kappa)$-robust 聚合器（定义 10.5），则 $\mathcal A\circ\mathrm{ARC}$ 是 $(\hat B,\tilde\kappa)$-robust 的，
> $$
> \tilde\kappa=\kappa+\frac{2\hat B}{N-2\hat B}\ \le\ 3\kappa ,
> \tag{10.3}
> $$
> 末步用 $(\hat B,\kappa)$-robustness 的必要条件 $\kappa\ge\hat B/(N-2\hat B)$。

> **定义 10.5（标准 $(f,\kappa)$-robustness；Allouah et al. 2023）** $\mathcal A:\mathbb R^{N\times D}\to\mathbb R^D$ 是 $(\hat B,\kappa)$-robust 的，若对任意输入与任意 $S\subseteq[N]$、$|S|=N-\hat B$，
> $$
> \big\|\mathcal A(x_1,\ldots,x_N)-\bar x_S\big\|^2\le\frac\kappa{|S|}\sum_{i\in S}\|x_i-\bar x_S\|^2 .
> \tag{10.4}
> $$

**与本文 A6 的接口。** 定义 10.5 是对**所有**大小 $N-\hat B$ 的子集量化的，而 A6 的 (3.5) 是关于**诚实集** $\mathcal H$（$|\mathcal H|=H=N-B\ge N-\hat B$）陈述的。命题 14.5（任意参照子集版）同时给出这两者：

* 取 $\mathcal G$ 为任意大小 $N-\hat B$ 的子集 $\Rightarrow$ Multi-Krum 满足**定义 10.5**，常数 $\kappa_q\le40$（推论 14.5a(b)），于是定理 10.4 可以直接引用；
* 取 $\mathcal G=\mathcal H$ $\Rightarrow$ 直接得到 A6 所需的形式。

二者合起来给出：

> **推论 10.7（A6 成立，且无残差）**
> 在 Algorithm 1' 中取 $\mathcal A_q=\mathrm{MK}_{s_q}\circ\mathrm{ARC}$。由推论 14.5a(b)，$\mathrm{MK}_{s_q}$ 是 $(\hat B,\kappa_q)$-robust 的（$\kappa_q\le40$）；再由定理 10.4，$\mathcal A_q$ 是 $(\hat B,\kappa_q^\star)$-robust 的，
> $$
> \boxed{\
> \kappa_q^\star=\kappa_q+\frac{2\hat B}{N-2\hat B}\ \le\ 41
> \quad(\hat B\le N/4,\ N\ge8,\ s_q=\lceil(N-\hat B)/2\rceil).\ }
> \tag{10.5}
> $$
> 从而 **A6 以 $\kappa_q^\star$ 成立**：
> $$
> \mathbb E\big\|z_t-\bar q_t\big\|^2\ \le\ \kappa_q^\star\,\mathbb EV_{q,t}
> \qquad\text{对每个 }t .
> \tag{10.6}
> $$
> **注意本版不能再往下写 $\le\kappa_q^\star\Sigma_\beta^2/p$**：定理 7.4 只给出 $V_{q,t}$ 的**时间平均**界（(7.6)），逐 $t$ 的 $\mathbb EV_{q,t}\le\Sigma_\beta^2/p$ 在本版**不成立**（$\varrho_t$ 含 $\beta^{2t}$ 预热项，$t$ 小时 $\mathbb EV_{u,t}$ 可以超过 $\Sigma_\beta^2$）。带 $1/p$ 的形式只在时间平均层面成立，见 (8.2)。下游的引理 11.7 正是这样使用的：(11.7) 逐 $t$ 只用 (10.6)，(11.8) 才做时间平均并代入 (7.6)。
> **裁剪的全部代价就是把 $\kappa_q$ 换成 $\kappa_q^\star$（一个加性的、不超过 $1$ 的项），不产生任何附加项。**

*证明.* 由推论 14.5a(b)，对任意 $S\subseteq[N]$、$|S|=N-\hat B$ 有 $\|\mathrm{MK}_{s_q}(x)-\bar x_S\|^2\le\frac{\kappa_q}{|S|}\sum_{i\in S}\|x_i-\bar x_S\|^2$，即定义 10.5。定理 10.4 给出 $\mathcal A_q=\mathrm{MK}_{s_q}\circ\mathrm{ARC}$ 的 $(\hat B,\kappa_q^\star)$-robustness。再取 $S\subseteq\mathcal H$、$|S|=N-\hat B$：由 $\hat B\le N/4$ 与 $\frac{2\hat B}{N-2\hat B}\le1$ 得 $\kappa_q^\star\le41$。最后由命题 14.5 取 $\mathcal G=\mathcal H$（$g=H$）直接得到关于 $\bar q_t$ 与 $V_{q,t}$ 的形式，其常数为 $\max\{\kappa(H),\kappa_q^\star\}$；在所给参数下 $\kappa(H)\le72$，故统一取 $\kappa_q^\star:=72+1=73$ 亦可。为叙述简洁，下文一律用记号 $\kappa_q^\star$，其在实例化下满足 $\kappa_q^\star\le73=O(1)$。$\square$

> **注 10.7a（关于 $\mathcal G=\mathcal H$ 与 $\mathcal G$ 为 $N-\hat B$ 子集两个常数的取舍）**
> 严格说，定理 10.4 输出的是**定义 10.5 意义**下的鲁棒性（参照子集大小恰为 $N-\hat B$），而 A6 需要的是关于 $\mathcal H$ 的形式。二者在 $B=\hat B$ 时重合。$B<\hat B$ 时有两种处理：
> **(i)** 用引理 10.6 的归约（下），常数为 $2H(\kappa_q^\star+1)/(N-\hat B)$；
> **(ii)** 直接对 $\mathcal A_q=\mathrm{MK}\circ\mathrm{ARC}$ 重跑命题 14.5 的四步论证并取 $\mathcal G=\mathcal H$——由于 ARC 只是把全体输入按同一标量缩放、不改变 Step 1–4 中任何一步的计数结构，这一路径给出常数 $\kappa(H)+\frac{2\hat B}{N-2\hat B}\le73$。
> 本文采用 (ii)，故全文的 $\kappa_q^\star\le73$。（若只关心 $B=\hat B$ 的最坏情形，则 $\kappa_q^\star\le41$。）

> **引理 10.6（备用归约，本文不再使用）** 设 $\mathcal A\circ\mathrm{ARC}$ 为定义 10.5 意义下的 $(\hat B,\tilde\kappa)$-robust，$B\le\hat B$，$s_\star:=N-\hat B\le H$。则
> $$
> \big\|z_t-\bar q_t\big\|^2\ \le\ \frac{2H(\tilde\kappa+1)}{N-\hat B}\,V_{q,t} .
> \tag{10.5'}
> $$
> *证明.* 取 $S\subseteq\mathcal H$、$|S|=s_\star$。由定义 10.5 与 $\bar q_t^S$ 的最小二乘中心性质，$\|z_t-\bar q_t^S\|^2\le\frac{\tilde\kappa}{s_\star}\sum_{i\in S}\|q_t^{(i)}-\bar q_t\|^2\le\frac{\tilde\kappa H}{s_\star}V_{q,t}$；又由 (J)，$\|\bar q_t^S-\bar q_t\|^2\le\frac H{s_\star}V_{q,t}$。两式用 (Y)（$\nu=1$）合并。$\square$
> 该归约在 $\hat B\le N/4$ 下给出 $\le\frac83(\tilde\kappa+1)$，比路径 (ii) 松，故只作备用。

### 10.3 输出有界性（只用于讨论，不进入主定理）

> **引理 10.8（Bounded Output；Allouah et al. 2025, Lem 5.1）** 若 $\mathcal A$ 满足 $\|\mathcal A(x_1,\ldots,x_N)\|\le\max_{i\in[N]}\|x_i\|$（Multi-Krum 输出为子集平均，自动满足），则
> $$
> \big\|\mathcal A\circ\mathrm{ARC}(x_1,\ldots,x_N)\big\|\ \le\ \max_{i\in S}\|x_i\|
> \qquad\forall S,\ |S|=N-\hat B .
> \tag{10.7}
> $$
> 取 $S\subseteq\mathcal H$ 得：$\|z_t\|\le\max_{i\in\mathcal H}\|q_t^{(i)}\|$ **几乎必然**。

*证明.* 由引理 10.2(b)，$\max_{i\in S}\|x_i\|\ge C^{\mathrm{arc}}$；而裁剪后全体输入的范数都 $\le C^{\mathrm{arc}}$，再用 $\mathcal A$ 的极大范数性质。$\square$

> **注 10.9（(10.7) 的两个作用）** 引理 10.8 **不进入主定理的任何常数**，但有两个不可替代的用途。
> **(a) 先验有限性（必需）**：引理 11.0' 需要 $Z_t=\mathbb E\|z_t\|^2<\infty$，而 $z_t$ 是含**任意恶意** Byzantine 输入的聚合输出；(10.7) 把 $\|z_t\|$ 卡在**纯诚实量** $\max_{i\in\mathcal H}\|q_t^{(i)}\|$ 之下，从而把 Byzantine 从有限性论证中彻底剔除。若无这条性质，定理 11.15 的两次移项无从合法化。
> **(b) cap 的实践判据**：见 §13.2——cap 只有在**诚实方 tracker 范数的最大值超过 $C_{\max}$** 时才可能起作用，而这是一个纯诚实量的尾部事件，与 Byzantine 行为无关。这为 $C_{\max}$ 的实际取值提供了可监测的判据。

> **注 10.10（为什么 cap 不能并入 ARC 的阈值）** 一个自然的想法是把第 11 步的 cap 直接写进 (10.2)，即用 $\min\{C^{\mathrm{arc}},C_{\max}\}$ 做预聚合裁剪。**这会破坏定理 10.4**：Allouah et al. (2025) 的 Lemma 3.1 证明，对**任何固定的** $C$ 与**任何** $\kappa'$，$\mathcal A\circ\mathrm{Clip}_C$ 都不是 $(\hat B,\kappa')$-robust；在 $C_{\max}$ 起作用的区间内 $\min\{C^{\mathrm{arc}},C_{\max}\}=C_{\max}$ 即为静态裁剪。因此 **cap 必须留在聚合之后，且（见 §11）只作用于二阶矩累加器**。

---

## 11. Adam：Preconditioner、动量偏差、Estimator 误差与自洽闭合

本节相对变形A 的改动集中在两处：

* **新增 §11.3（动量偏差）**：客户端动量缓存 $\bar u_t$ 不是 $h_t=\nabla f(\theta_t)$ 的无偏估计，二者之差 $\mathcal B_t$ 必须显式闭合。这是引入动量的唯一新代价。
* **§11.5（漂移）重写**：驱动项中的 $\sigma^2/H$ 获得因子 $(1-\beta)^2$。这是引入动量的全部收益来源。

其余与变形A 逐字相同的是：引理 11.1（preconditioner）、引理 11.3（一阶矩的二阶矩）、定理 11.12（服务器动量残差）、引理 11.13 与 11.14（单步下降与求和形式）。**注意引理 11.7 与引理 11.9 恰恰是改过的两条**（前者末项由 $3\sigma^2/H$ 换成 $3\mathcal B_t$，后者是漂移的重写）。

**本节记号。** 除变形A 已有的
$$
\Delta_{\mathrm{est},t}:=\mathbb E\|z_t-h_t\|^2,\quad
Z_t:=\mathbb E\|z_t\|^2,\quad
R_{\mathrm{mom},t}:=\mathbb E\|m_{t+1}-z_t\|^2,
\tag{11.0}
$$
$$
G_T:=\frac1T\sum_{t=0}^{T-1}\mathbb E\|h_t\|^2,\quad
\bar Z:=\frac1T\sum_{t=0}^{T-1}Z_t,\quad
\bar\Delta_{\mathrm{est}}:=\frac1T\sum_{t=0}^{T-1}\Delta_{\mathrm{est},t},\quad
\bar R_{\mathrm{mom}}:=\frac1T\sum_{t=0}^{T-1}R_{\mathrm{mom},t}
\tag{11.0'}
$$
（$\bar E=\bar E^{(T)}$ 见 (9.13)，$\bar D^{(T)}$ 的范围是 $t=1..T-1$，其余横杠量一律为 $t=0..T-1$ 的平均）
之外，新增**客户端动量偏差**
$$
\mathcal B_t:=\mathbb E\big\|\bar u_t-h_t\big\|^2,
\qquad
\bar{\mathcal B}:=\frac1T\sum_{t=0}^{T-1}\mathcal B_t ,
\tag{11.0a}
$$
以及两个动量常数
$$
c_\beta:=\frac{\beta_1^2}{(1-\beta_1)^2}\ (\text{服务器 Adam}),
\qquad
c_\beta^{\mathrm{cl}}:=\frac{\beta^2}{(1-\beta)^2}\ (\text{客户端}).
\tag{11.0b}
$$

### 11.1 Preconditioner 的确定性界（无需任何假设）

> **引理 11.1** 令 $\hat g_t=\mathrm{clip}_{C_{\max}}(z_t)$、$D_t^A:=\operatorname{diag}\big((\sqrt{v_{t+1}}+\sqrt\epsilon)^{-1}\big)$。则确定性地 $\|\hat g_t\|\le C_{\max}$、$0\le v_{t+1,j}\le C_{\max}^2$，从而
> $$
> \lambda_AI\preceq D_t^A\preceq\Lambda_AI,
> \qquad
> \lambda_A=\frac1{C_{\max}+\sqrt\epsilon},
> \quad
> \Lambda_A=\frac1{\sqrt\epsilon} .
> \tag{11.2}
> $$

*证明.* 逐坐标 $\tilde v_{t+1,j}=(1-\beta_2)\sum_{\tau\le t}\beta_2^{t-\tau}\hat g_{\tau,j}^2\le(1-\beta_2^{t+1})C_{\max}^2$，除以 bias correction 因子得 $v_{t+1,j}\le C_{\max}^2$（精确，与 $t$ 无关）。$\square$

> **注 11.2** 与变形A 相同：本引理**只用到喂进 $\tilde v$ 的向量有界**，而 $\hat g_t$ 只出现在二阶矩累加器里。$\|m_{t+1}\|$ 与位移仍无确定性上界，由引理 11.3 的期望界代替。

**正因为本版放弃了一切确定性界，下面这条先验有限性引理是必需的**——全文有六处形如「$X\le a+bX\Rightarrow X\le\frac a{1-b}$」的移项（(7.6)、(11.6)、引理 9.7、定理 11.12、定理 11.15 的 Step 2 与 Step 3），而该推理只在 $X<\infty$ 时成立。

> **引理 11.0'（先验有限性）** 在 A1–A3 下，对任意固定的 $\eta>0$、$T<\infty$ 与一切 $0\le t<T$：
> $$
> \mathbb E\|u_t^{(i)}\|^2<\infty\ (i\in\mathcal H),\quad
> \mathbb E\|q_t^{(i)}\|^2<\infty\ (i\in\mathcal H),\quad
> Z_t<\infty,\quad
> \mathbb E\|m_{t+1}\|^2<\infty,\quad
> \mathbb E\|h_t\|^2<\infty .
> \tag{11.1}
> $$
> 从而 $\bar Z,G_T,\bar\Delta_{\mathrm{est}},\bar E,\bar{\mathcal B},\bar R_{\mathrm{mom}},\bar V_q^{(T)},\bar V_r^{(T)}$ 全部有限，上述六处移项均合法。

*证明.* 对 $t$ 作有限归纳。$t=0$：$\theta_0$ 确定，由 A1（每个 $f_i$ 为 $L$-smooth，故 $\|\nabla f_i(\theta_0)\|<\infty$）与 A3 得 $\mathbb E\|g_0^{(i)}\|^2\le\sigma^2+\|\nabla f_i(\theta_0)\|^2<\infty$，而 $u_0^{(i)}=g_0^{(i)}$、$q_0^{(i)}=(I-Q_0)\cdot0+Q_0u_0^{(i)}$ 故 $\mathbb E\|q_0^{(i)}\|^2\le\mathbb E\|u_0^{(i)}\|^2<\infty$。

设结论对 $0,\ldots,t-1$ 成立。

**(i) $Z_{t-1}<\infty$（Byzantine 在此被完全剔除）.** 由**引理 10.8**，$\|z_{t-1}\|\le\max_{i\in\mathcal H}\|q_{t-1}^{(i)}\|$ 几乎必然，故
$$
Z_{t-1}=\mathbb E\|z_{t-1}\|^2\le\sum_{i\in\mathcal H}\mathbb E\|q_{t-1}^{(i)}\|^2<\infty .
$$
这是本证明的关键一步：ARC 的输出有界性把**任意恶意的** Byzantine 输入从有限性论证中彻底排除；若无这条性质，$z_{t-1}$ 可以由 Byzantine 任意放大，$Z_{t-1}$ 的有限性无从谈起。

**(ii) 迭代点与梯度.** 由引理 11.3，$\mathbb E\|m_t\|^2\le\max_{\tau<t}Z_\tau<\infty$，故 $\mathbb E\|\theta_t-\theta_{t-1}\|^2\le\eta^2\Lambda_A^2\mathbb E\|m_t\|^2<\infty$，累加得 $\mathbb E\|\theta_t-\theta_0\|^2<\infty$。再由 A1 的逐客户端光滑性，
$$
\mathbb E\|\nabla f_i(\theta_t)\|^2\le2\|\nabla f_i(\theta_0)\|^2+2L^2\mathbb E\|\theta_t-\theta_0\|^2<\infty,
$$
从而 $\mathbb E\|h_t\|^2<\infty$，并由 A3 得 $\mathbb E\|g_t^{(i)}\|^2<\infty$。

**(iii) 缓存与 tracker.** 由 (7.0) 的凸权重与 Jensen，$\mathbb E\|u_t^{(i)}\|^2\le\max_{\tau\le t}\mathbb E\|g_\tau^{(i)}\|^2<\infty$。由 (2.6) 与 $Q_t$ 的正交投影性质，$\|q_t^{(i)}\|^2=\|(I-Q_t)q_{t-1}^{(i)}\|^2+\|Q_tu_t^{(i)}\|^2\le\|q_{t-1}^{(i)}\|^2+\|u_t^{(i)}\|^2$，取期望即得有限。归纳完成。

由于 $T<\infty$，全部时间平均都是有限项之和，故有限。$\square$

> **注 11.2a（引理 10.8 的真正用途）** §10.3 曾标注"主定理不使用引理 10.8"。这一说法应更正为：**引理 10.8 不进入主定理的任何常数，但它是引理 11.0' 的关键前提**，因而是 (11.19) 成立的必要环节。ARC 的这条输出有界性在本版中承担两个互不相干的角色——(a) §13.2 中判断 cap 何时被触发的可监测判据；(b) 这里把 Byzantine 从有限性论证中剔除。

### 11.2 一阶矩的二阶矩

> **引理 11.3** 对一切 $t\ge0$，$\mathbb E\|m_{t+1}\|^2\le\sum_{\tau=0}^t(1-\beta_1)\beta_1^{t-\tau}Z_\tau$，从而
> $$
> \frac1T\sum_{t=0}^{T-1}\mathbb E\|m_{t+1}\|^2\le\bar Z,
> \qquad
> \frac1T\sum_{t=1}^{T-1}\mathbb E\|m_t\|^2\le\bar Z ,
> \tag{11.4}
> $$
> 且 $\mathbb E\|\theta_{t+1}-\theta_t\|^2\le\eta^2\Lambda_A^2\,\mathbb E\|m_{t+1}\|^2$。

*证明.* $m_{t+1}=\sum_\tau w_{t,\tau}z_\tau$，$w_{t,\tau}=(1-\beta_1)\beta_1^{t-\tau}$，$W_t=\sum_\tau w_{t,\tau}\le1$。由 Jensen 与 $W_t\le1$ 得 $\|m_{t+1}\|^2\le\sum_\tau w_{t,\tau}\|z_\tau\|^2$；对 $t$ 求和并交换次序（$\sum_{t\ge\tau}(1-\beta_1)\beta_1^{t-\tau}\le1$）即得。$\square$

### 11.3 客户端动量偏差（本版新增）

> **引理 11.4（动量偏差的递推）** 在 A1–A3 下，$\mathcal B_0\le\sigma^2/H$，且对 $t\ge1$
> $$
> \mathcal B_t\ \le\ \beta\,\mathcal B_{t-1}
> +\frac{\beta^2}{1-\beta}L^2\,\mathbb E\|\theta_t-\theta_{t-1}\|^2
> +(1-\beta)^2\frac{\sigma^2}H .
> \tag{11.5}
> $$

*证明.* 由 (2.1a) 对 honest 取平均得 $\bar u_t=\beta\bar u_{t-1}+(1-\beta)\bar g_t$（$t\ge1$），故

$$
\bar u_t-h_t
=\beta\big(\bar u_{t-1}-h_{t-1}\big)+\beta\big(h_{t-1}-h_t\big)+(1-\beta)\big(\bar g_t-h_t\big).
$$

前两项为 $\mathcal F_t$-可测，而 $\mathbb E[\bar g_t-h_t\mid\mathcal F_t]=0$（A2），故交叉项期望为零：

$$
\mathcal B_t=\mathbb E\big\|\beta(\bar u_{t-1}-h_{t-1})+\beta(h_{t-1}-h_t)\big\|^2
+(1-\beta)^2\mathbb E\|\bar g_t-h_t\|^2 .
$$

第二项由 (3.1) 不超过 $(1-\beta)^2\sigma^2/H$。第一项（$\beta>0$ 时）用 (Y) 取 $\nu=\frac{1-\beta}\beta$（使 $\beta^2(1+\nu)=\beta$、$\beta^2(1+\nu^{-1})=\frac{\beta^2}{1-\beta}$，两式为**恒等式**，故 (11.5) 的系数是最优的）并用 A1 的 $\|h_t-h_{t-1}\|\le L\|\theta_t-\theta_{t-1}\|$；$\beta=0$ 时该项为零，(11.5) 平凡成立（此时算法退化为变形A）。$t=0$ 时 $u_0^{(i)}=g_0^{(i)}$ 故 $\bar u_0=\bar g_0$，由 (3.1) 得 $\mathcal B_0\le\sigma^2/H$。$\square$

> **推论 11.5（动量偏差的时间平均）**
> $$
> \boxed{\
> \bar{\mathcal B}\ \le\ (1-\beta)\frac{\sigma^2}H
> +c_\beta^{\mathrm{cl}}L^2\eta^2\Lambda_A^2\,\bar Z
> +\frac{\sigma^2}{H(1-\beta)T}\ } .
> \tag{11.6}
> $$

*证明.* 对 (11.5) 从 $t=1$ 到 $T-1$ 求和，两边加 $\mathcal B_0$ 并把右端的 $\sum_{t=0}^{T-2}$ 放大为 $\sum_{t=0}^{T-1}$：

$$
\sum_{t=0}^{T-1}\mathcal B_t\le\frac{\sigma^2}H+\beta\sum_{t=0}^{T-1}\mathcal B_t
+\frac{\beta^2L^2}{1-\beta}\sum_{t=1}^{T-1}\mathbb E\|\theta_t-\theta_{t-1}\|^2
+T(1-\beta)^2\frac{\sigma^2}H .
$$

由引理 11.3，$\sum_{t=1}^{T-1}\mathbb E\|\theta_t-\theta_{t-1}\|^2\le\eta^2\Lambda_A^2\sum_{t=1}^{T-1}\mathbb E\|m_t\|^2\le T\eta^2\Lambda_A^2\bar Z$。移项、除以 $(1-\beta)T$ 即得。$\square$

> **注 11.6（动量偏差的两个成分与 $\beta$ 的作用方向）** (11.6) 右端三项的性质完全不同：
> * $(1-\beta)\sigma^2/H$：随 $\beta\to1$ **减小**；
> * $c_\beta^{\mathrm{cl}}L^2\eta^2\Lambda_A^2\bar Z$：随 $\beta\to1$ **增大**（$c_\beta^{\mathrm{cl}}=\beta^2/(1-\beta)^2\to\infty$），但带因子 $\eta^2$；
> * $O(1/T)$ 项：预热代价，随 $\beta\to1$ 增大但随 $T$ 消失。
>
> 第二项是引入客户端动量的**全部代价**：它在 §11.8 的自洽闭合中要求步长满足 $\eta\lesssim(1-\beta)$（注 11.16）。由于最终取 $\eta=\Theta(T^{-1/2})$，这意味着 $1-\beta$ 也可以取到 $\Theta(T^{-1/2})$——恰好使第一项与优化项同阶（推论 12.4）。

### 11.4 Estimator 误差

> **引理 11.7** 在 A6（以 $\kappa_q^\star$，推论 10.7）下，
> $$
> \Delta_{\mathrm{est},t}\ \le\ 3\kappa_q^\star\,\mathbb EV_{q,t}+3E_t+3\mathcal B_t ,
> \tag{11.7}
> $$
> 从而由定理 7.4，
> $$
> \bar\Delta_{\mathrm{est}}\ \le\ \frac{3\kappa_q^\star}p\Sigma_\beta^2+3\bar E+3\bar{\mathcal B}+O(1/T) .
> \tag{11.8}
> $$
> 又 $Z_t\le2\mathbb E\|h_t\|^2+2\Delta_{\mathrm{est},t}$，故
> $$
> \bar Z\ \le\ 2G_T+2\bar\Delta_{\mathrm{est}} .
> \tag{11.9}
> $$

*证明.* $z_t-h_t=(z_t-\bar q_t)+(\bar q_t-\bar u_t)+(\bar u_t-h_t)=(z_t-\bar q_t)+e_t+(\bar u_t-h_t)$，三项 Young（(Y) 取 $\nu=2$ 两次，即 $\|a+b+c\|^2\le3(\|a\|^2+\|b\|^2+\|c\|^2)$）后分别代入 (10.6)、$E_t$ 的定义与 (11.0a)。$\square$

> **注 11.8（与变形A 的差别）** 变形A 的对应式是
> $\Delta_{\mathrm{est},t}\le3\kappa_q^\star\mathbb EV_{q,t}+3E_t+3\sigma^2/H$，
> 末项是**采样噪声**；本版末项换成**动量偏差** $\mathcal B_t$，其时间平均的主项是 $(1-\beta)\sigma^2/H$。这是「用一个可以被 $\beta$ 压小的项，换掉一个不能被压小的项」。
> 与完整版相比，(11.7) 中还**没有** clipping 残差项 $R_{\mathrm{clip},t}$——那是变形A 把 cap 移到二阶矩路径换来的（注 11.8a）。

> **注 11.8a（为什么 cap 必须只作用于二阶矩：一个反例）** 若像原完整版那样让裁剪后的 $\hat g_t$ 同时进入一阶矩，则误差分解中出现残差 $\mathbb E\big(\|z_t\|-C_{\max}\big)_+^2$。**仅有 A3（期望意义的有界方差）时，该量的上确界与 $C_{\max}$ 无关**：取 $X$ 以概率 $m_2/A^2$ 取值 $A$、否则取 $0$，则 $\mathbb EX^2=m_2$ 而
> $$
> \mathbb E\big(X-C\big)_+^2=\frac{(A-C)^2}{A^2}m_2\ \xrightarrow[A\to\infty]{}\ m_2
> \qquad\text{对任意 }C ,
> $$
> 故 $\sup\mathbb E(X-C)_+^2=m_2$，**不随 $C$ 衰减**。相比之下**一次量**满足 $\mathbb E(X-C)_+\le\frac{m_2}{4C}$（由恒等式 $(X-C)_+\le\frac{X^2}{4C}$，在 $A=2C$ 处取等），随 $C$ 衰减。
> 由此得两个结论：(i) 在 $\frac1T\sum\mathbb E\|\nabla f\|^2$ 型结论中，任何误差项都会被 Young 或 Cauchy–Schwarz 平方一次，因此完整版的结构下该残差**只能靠四阶矩或几乎必然有界性闭合**——其可选条件 A10'（有界随机梯度）不是「分析不够好」，而是必然；(ii) Koloskova et al. (2023) 给出的裁剪偏差紧界 $\min\{\sigma,\sigma^2/c\}$ 只存在于**一次幂、$\min_t$** 形式（其 Thm 3.3），与本文献线的平方型结论不可直接比较。
> **变形A 及本版绕开这一障碍的方式不是「更好地界住该项」，而是让它根本不出现**：更新方向上没有裁剪算子（算法第 12 步），$\hat g_t$ 只喂给 $\tilde v$。

### 11.5 Honest 均值动量漂移（本版重写）

> **引理 11.9** 对 $t\ge1$，
> $$
> D_t=\mathbb E\big\|\bar u_t-\bar u_{t-1}\big\|^2
> \ \le\ 3(1-\beta)^2\Big[\frac{\sigma^2}H+L^2\mathbb E\|\theta_t-\theta_{t-1}\|^2+\mathcal B_{t-1}\Big] ,
> \tag{11.10}
> $$
> 从而
> $$
> \boxed{\
> \bar D^{(T)}:=\frac1T\sum_{t=1}^{T-1}D_t
> \ \le\ 6(1-\beta)^2\frac{\sigma^2}H+3L^2\eta^2\Lambda_A^2\,\bar Z+O(1/T)\ } .
> \tag{11.11}
> $$

*证明.* 由 $\bar u_t=\beta\bar u_{t-1}+(1-\beta)\bar g_t$ 得 $\bar u_t-\bar u_{t-1}=(1-\beta)\big(\bar g_t-\bar u_{t-1}\big)$，而

$$
\bar g_t-\bar u_{t-1}=\big(\bar g_t-h_t\big)+\big(h_t-h_{t-1}\big)-\big(\bar u_{t-1}-h_{t-1}\big),
$$

三项 Young 并用 (3.1)、A1 与 (11.0a) 即得 (11.10)。取时间平均并代入引理 11.3 与推论 11.5：

$$
\bar D^{(T)}\le3(1-\beta)^2\Big[\frac{\sigma^2}H+L^2\eta^2\Lambda_A^2\bar Z
+(1-\beta)\frac{\sigma^2}H+c_\beta^{\mathrm{cl}}L^2\eta^2\Lambda_A^2\bar Z\Big]+O(1/T).
$$

用 $(1-\beta)^2\big(1+(1-\beta)\big)\le2(1-\beta)^2$ 与
$$
(1-\beta)^2\big(1+c_\beta^{\mathrm{cl}}\big)=(1-\beta)^2+\beta^2\le1
$$
即得 (11.11)。$\square$

> **注 11.10（这一条就是全部收益）** 对照变形A 的引理 11.7：
> $$
> \text{变形A：}\quad\bar D^{(T)}\le2L^2\eta^2\Lambda_A^2\bar Z+\frac{4\sigma^2}H,
> \qquad
> \text{本版：}\quad\bar D^{(T)}\le3L^2\eta^2\Lambda_A^2\bar Z+\frac{6(1-\beta)^2\sigma^2}H .
> $$
> $\eta^2$ 项的系数从 $2$ 变成 $3$（无关紧要），而 $\sigma^2/H$ 项的系数从 $4$ 变成 $6(1-\beta)^2$：$\beta=0.9$ 时是 $0.06$（缩小约 $67$ 倍），$\beta=0.99$ 时是 $6\times10^{-4}$（缩小约 $6700$ 倍）。
> 由于 $\bar D^{(T)}$ 经 (9.14) 以 $4/c^2$ 的系数进入 $\bar E$、再进入 error floor，这个缩小是**直接的、一比一的**。

> **推论 11.11（tracking error 的时间平均）** 把 (11.11) 代入 (9.14)：
> $$
> \bar E\ \le\ \frac{2E_0}{cT}+\frac{24(1-\beta)^2\sigma^2}{Hc^2}
> +\frac{12L^2\eta^2\Lambda_A^2}{c^2}\bar Z+\Xi_c+O(1/T),
> \tag{11.12}
> $$
> 其中**定理 A 分支**取 $c=p$、$\Xi_p=0$；**定理 B 分支**取 $c=\chi$、$\Xi_\chi=\dfrac{20\kappa_r\bar V_r^{(T)}}{\chi\,\omega_{\mathrm{top}}}$，$\bar V_r^{(T)}\le(1+p^{-1/2})^2\Sigma_\beta^2+O(1/T)$。

### 11.6 服务器动量残差

> **定理 11.12** $\displaystyle\bar R_{\mathrm{mom}}\le6c_\beta\bar\Delta_{\mathrm{est}}+3c_\beta L^2\eta^2\Lambda_A^2\bar Z+\frac{\beta_1^2Z_0}{(1-\beta_1)T}$。

*证明.* 与变形A 定理 11.9 逐字相同（该推导只用到 $m_{t+1}=\beta_1m_t+(1-\beta_1)z_t$、A1 与引理 11.3，与客户端动量无关）。$\square$

### 11.7 Adam 单步下降

> **引理 11.13** 记 $d_t:=D_t^Am_{t+1}$、$a_t:=m_{t+1}-h_t$。则 $\mathbb E\|a_t\|^2\le2R_{\mathrm{mom},t}+2\Delta_{\mathrm{est},t}$，
> $$
> \mathbb E\langle h_t,d_t\rangle\ge\frac{\lambda_A}2\mathbb E\|h_t\|^2-\frac{\Lambda_A^2}{\lambda_A}\big(R_{\mathrm{mom},t}+\Delta_{\mathrm{est},t}\big),
> \qquad
> \mathbb E\|d_t\|^2\le\Lambda_A^2\mathbb E\|m_{t+1}\|^2 ,
> \tag{11.13}
> $$
> 从而**对任意 $\eta>0$**
> $$
> \mathbb Ef(\theta_{t+1})\le\mathbb Ef(\theta_t)-\frac{\eta\lambda_A}2\mathbb E\|h_t\|^2
> +\eta\frac{\Lambda_A^2}{\lambda_A}\big(R_{\mathrm{mom},t}+\Delta_{\mathrm{est},t}\big)
> +\frac{L\eta^2\Lambda_A^2}2\mathbb E\|m_{t+1}\|^2 .
> \tag{11.14}
> $$

*证明.* 与变形A 的引理 11.11 逐字相同（该推导只用引理 11.1 的谱界与 (11.4)，与客户端动量无关）。$\square$

求和、用 $\mathbb Ef(\theta_T)\ge f^\star$、除以 $\eta\lambda_AT/2$ 并代入 (11.4)：

> **引理 11.14（求和形式）** 记 $K:=\dfrac{2\Lambda_A^2}{\lambda_A^2}$。则
> $$
> G_T\ \le\ \frac{2\Delta_0}{\eta\lambda_AT}+K\big(\bar R_{\mathrm{mom}}+\bar\Delta_{\mathrm{est}}\big)
> +\frac{L\eta\Lambda_A^2}{\lambda_A}\,\bar Z .
> \tag{11.15}
> $$

### 11.8 自洽闭合

> **定理 11.15（闭合）** 记 $c\in\{p,\chi\}$，
> $$
> \Phi_c:=\frac{3\kappa_q^\star\Sigma_\beta^2}p
> +\frac{3(1-\beta)\sigma^2}H
> +\frac{72(1-\beta)^2\sigma^2}{Hc^2}
> +3\Xi_c ,
> \tag{11.16}
> $$
> $$
> \Psi_c:=\frac{12}{c^2}+c_\beta^{\mathrm{cl}},
> \qquad
> \theta_c:=3\Psi_cL^2\eta^2\Lambda_A^2 .
> \tag{11.17}
> $$
> 若步长满足
> $$
> \boxed{\
> \eta\ \le\ \eta_{\max}:=\frac{\lambda_A}{24L\Lambda_A^2}\cdot
> \min\Big\{1,\ \frac1{\sqrt{\Psi_c\big(1+6c_\beta\big)}},\ \frac1{\sqrt{c_\beta}}\Big\}\ }
> \tag{11.18}
> $$
> 则
> $$
> \boxed{\
> G_T\ \le\ \frac{4\Delta_0}{\eta\lambda_AT}+C_A^\sharp\,\Phi_c+\frac{\Xi_0^\sharp}T\ },
> \qquad
> C_A^\sharp:=2\Big[K\big(1+6c_\beta\big)+\tfrac12\Big]
> =\frac{4\Lambda_A^2(1+6c_\beta)}{\lambda_A^2}+1 ,
> \tag{11.19}
> $$
> 其中
> $$
> \Xi_0^\sharp:=C_A^\sharp\,\mathcal O_1+\frac{2K\beta_1^2Z_0}{1-\beta_1}
> \tag{11.23}
> $$
> 由 (11.20') 完全显式。**$\Xi_0^\sharp$ 不含 $T$，但依赖 $\beta$**，形如 $\Xi_a+\frac{\Xi_b}{1-\beta}$；当 $\beta$ 随 $T$ 取值（推论 12.4）时须回到 (11.20') 的显式形式来判定 $\Xi_0^\sharp/T$ 的阶。

*证明.* 分三步。

**Step 1（消去 $\bar E$ 与 $\bar{\mathcal B}$）.** 把 (11.12) 与 (11.6) 代入 (11.8)：

$$
\bar\Delta_{\mathrm{est}}
\le\frac{3\kappa_q^\star\Sigma_\beta^2}p
+3\Big[\frac{24(1-\beta)^2\sigma^2}{Hc^2}+\frac{12L^2\eta^2\Lambda_A^2}{c^2}\bar Z+\Xi_c\Big]
+3\Big[\frac{(1-\beta)\sigma^2}H+c_\beta^{\mathrm{cl}}L^2\eta^2\Lambda_A^2\bar Z\Big]
+\frac{\mathcal O_1}T,
$$

即

$$
\bar\Delta_{\mathrm{est}}\ \le\ \Phi_c+\theta_c\,\bar Z+\frac{\mathcal O_1}T ,
\tag{11.20}
$$

其中 $\theta_c=3\big(\tfrac{12}{c^2}+c_\beta^{\mathrm{cl}}\big)L^2\eta^2\Lambda_A^2$ 恰为 (11.17)，而 $\mathcal O_1$ **完全显式**：

$$
\mathcal O_1:=
\underbrace{\frac{6E_0}c}_{3\times(11.12)\text{ 的 }\frac{2E_0}{cT}}
+\underbrace{\frac{3\sigma^2}{H(1-\beta)}}_{3\times(11.6)\text{ 的预热项}}
+\underbrace{\frac{6\kappa_q^\star\sigma^2}{p(1-\beta^2)}}_{3\kappa_q^\star\times(7.6)\text{ 的暂态}}
+\underbrace{\frac{36(1-\beta)\sigma^2}{Hc^2}}_{3\times\frac4{c^2}\times(11.11)\text{ 的暂态}}
+\underbrace{\frac{120\,\kappa_r(1+p^{-1/2})^2\sigma^2}{\chi\,\omega_{\mathrm{top}}(1-\beta^2)}\mathbf 1\{c=\chi\}}_{3\times(11.12)\text{ 的 }\Xi_\chi\text{ 中 }(7.8)\text{ 的暂态}} .
\tag{11.20'}
$$

> **注 11.15a（$\mathcal O_1$ 依赖 $\beta$，不是绝对常数）** (11.20') 的第二、三、五项含 $\frac1{1-\beta}$ 或 $\frac1{1-\beta^2}$。因此本节及 §11 各处写的 $O(1/T)$ 精确说是 $O\big(\frac1{(1-\beta)T}\big)$：$\beta$ **固定**时它确实是 $O(1/T)$；但在推论 12.4 取 $1-\beta=\Theta(T^{-1/2})$ 时它是 $\Theta(T^{-1/2})$，与主项同阶（推论 12.4 的证明已按此处理）。

**Step 2（解出 $\bar Z$）.** 代入 (11.9)：$\bar Z\le2G_T+2\Phi_c+2\theta_c\bar Z+2\mathcal O_1/T$。由 (11.18)，

$$
\eta\le\frac{\lambda_A}{24L\Lambda_A^2\sqrt{\Psi_c(1+6c_\beta)}}\le\frac1{24L\Lambda_A\sqrt{\Psi_c}}
\quad\Longrightarrow\quad
\theta_c=3\Psi_cL^2\eta^2\Lambda_A^2\le\frac3{576}=\frac1{192}\le\frac14 ,
$$

（用 $\lambda_A\le\Lambda_A$ 与 $1+6c_\beta\ge1$），故 $2\theta_c\le\frac12$，于是

$$
\bar Z\ \le\ 4G_T+4\Phi_c+\frac{4\mathcal O_1}T .
\tag{11.21}
$$

**Step 3（解出 $G_T$）.** 把定理 11.12 与 (11.20) 代入 (11.15)：

$$
G_T\le\frac{2\Delta_0}{\eta\lambda_AT}+K(1+6c_\beta)\Phi_c+\mathcal A\,\bar Z
+\frac{K(1+6c_\beta)\mathcal O_1}T+\frac{K\beta_1^2Z_0}{(1-\beta_1)T},
$$
$$
\mathcal A:=\theta_cK\big(1+6c_\beta\big)+3c_\beta KL^2\eta^2\Lambda_A^2+\frac{L\eta\Lambda_A^2}{\lambda_A}.
$$

由 (11.18) 逐项验证：

* $\dfrac{L\eta\Lambda_A^2}{\lambda_A}\le\dfrac1{24}$，直接由 $\eta\le\frac{\lambda_A}{24L\Lambda_A^2}$；
* $\theta_cK(1+6c_\beta)=\dfrac{6\Psi_c(1+6c_\beta)L^2\eta^2\Lambda_A^4}{\lambda_A^2}$；代入 $\eta^2\le\dfrac{\lambda_A^2}{576\Psi_c(1+6c_\beta)L^2\Lambda_A^4}$ 得该项 $\le\dfrac6{576}=\dfrac1{96}$；
* $3c_\beta KL^2\eta^2\Lambda_A^2=\dfrac{6c_\beta L^2\eta^2\Lambda_A^4}{\lambda_A^2}$；代入 $\eta^2\le\dfrac{\lambda_A^2}{576c_\beta L^2\Lambda_A^4}$ 得该项 $\le\dfrac6{576}=\dfrac1{96}$。

故 $\mathcal A\le\frac1{24}+\frac1{96}+\frac1{96}=\frac1{16}<\frac18$。代入 (11.21)：

$$
(1-4\mathcal A)G_T\le\frac{2\Delta_0}{\eta\lambda_AT}+\big[K(1+6c_\beta)+4\mathcal A\big]\Phi_c
+\big[K(1+6c_\beta)+4\mathcal A\big]\frac{\mathcal O_1}T+\frac{K\beta_1^2Z_0}{(1-\beta_1)T}.
$$

由 $\mathcal A\le\frac1{16}$ 得 $1-4\mathcal A\ge\frac34$ 与 $4\mathcal A\le\frac14$；为陈述简洁，下面用更松的 $1-4\mathcal A\ge\frac12$、$4\mathcal A\le\frac12$，两边乘 $2$ 即得 (11.19) 与 (11.23)。$\square$

> **注 11.15b（常数尚有约 $1.5$ 倍余量）** 上一步若改用紧的 $1-4\mathcal A\ge\frac34$、$4\mathcal A\le\frac14$，(11.19) 可写成
> $$
> G_T\le\frac{8\Delta_0}{3\eta\lambda_AT}+\frac43\Big[K(1+6c_\beta)+\tfrac14\Big]\Phi_c+\frac{\Xi'_0}T ,
> $$
> 各常数改善约 $1.5$ 倍。独立数值复核（在 $\eta=\eta_{\max}$ 处精确求解 (11.6)(11.8)(11.9)(11.11)(11.12)(11.15) 与定理 11.12 构成的线性不等式组，$2\times10^4$ 组随机参数）显示 (11.19) 无违反、最差比值约 $0.52$，与此处的 $1.5$ 倍余量一致。

> **注 11.15c（(11.18) 的第三分支是多余的）** 由 $\Psi_c\ge\frac{12}{c^2}\ge12$（$c\le1$）得 $\Psi_c(1+6c_\beta)\ge12(1+6c_\beta)>c_\beta$，故 $\frac1{\sqrt{\Psi_c(1+6c_\beta)}}<\frac1{\sqrt{c_\beta}}$，第二分支恒紧于第三分支。(11.18) 可简化为
> $$
> \eta\le\frac{\lambda_A}{24L\Lambda_A^2}\cdot\min\Big\{1,\ \frac1{\sqrt{\Psi_c(1+6c_\beta)}}\Big\} .
> $$
> 正文保留三分支形式以使三条吸收条件与验证步骤一一对应。

> **注 11.16（步长条件的解读：$\eta\lesssim1-\beta$）** (11.18) 中 $\Psi_c=\frac{12}{c^2}+c_\beta^{\mathrm{cl}}$ 含客户端动量常数，故
> $$
> \eta_{\max}\ \asymp\ \frac{\lambda_A}{L\Lambda_A^2}\cdot\min\Big\{\frac{c}{\sqrt{1+6c_\beta}},\ \frac{1-\beta}{\beta\sqrt{1+6c_\beta}},\ \frac1{\sqrt{c_\beta}}\Big\} .
> $$
> 也就是说，除变形A 已有的 $\eta\lesssim c$ 之外，本版新增 $\boldsymbol{\eta\lesssim1-\beta}$。
> 这是客户端动量的**唯一**代价，来源是注 11.6 的第二项（动量缓存滞后于当前迭代点，滞后量正比于 $\frac\beta{1-\beta}\times$位移）。
> 由于最终取 $\eta=\Theta(T^{-1/2})$，该条件允许 $1-\beta=\Theta(T^{-1/2})$，此时 (11.16) 中**全部含 $\sigma^2$ 的项都退化为 $O(T^{-1/2})$**——见推论 12.4。

---

## 12. 主收敛定理

### 12.1 定理 A：基础版

> **定理 12.1（基础收敛保证）**
> 设 A1–A4、A6（以 $\kappa_q^\star$）、A8、A9 成立（**不需要 A5、A7**），步长满足 (11.18) 且取 $c=p$。则
> $$
> \boxed{
> \frac1T\sum_{t=0}^{T-1}\mathbb E\|\nabla f(\theta_t)\|^2
> \le
> \frac{4\Delta_0}{\eta\lambda_AT}
> +C_A^\sharp\bigg[
> \underbrace{\frac{6\kappa_q^\star\zeta^2}p}_{\text{(I) 异质性}}
> +\underbrace{\frac{6\kappa_q^\star(1-\beta)\sigma^2}{(1+\beta)p}
> +\frac{3(1-\beta)\sigma^2}H
> +\frac{72(1-\beta)^2\sigma^2}{Hp^2}}_{\text{(II) 随机噪声，全部带 }(1-\beta)\text{ 因子}}
> \bigg]
> +\frac{\Xi_0^\sharp}T }
> \tag{12.1}
> $$
> （已代入 $\Sigma_\beta^2=2\zeta^2+\frac{2(1-\beta)}{1+\beta}\sigma^2$）。

### 12.2 定理 B：完整算法版

> **定理 12.2** 设 A1–A9 成立、$\delta_{\mathrm{JL}}\le\omega_{\mathrm{top}}/4$、步长满足 (11.18) 且取 $c=\chi$。则 (12.1) 仍成立，只需作两处替换：**(a)** 把 $\frac{72(1-\beta)^2\sigma^2}{Hp^2}$ 中的 $p$ 换成 $\chi$（(I) 与 (II) 首项中的 $1/p$ **不变**，它们来自 $\bar V_q\le\Sigma_\beta^2/p$，恒为 $1/p$）；**(b)** 在 $C_A^\sharp[\cdot]$ 的**方括号内**额外加上
> $$
> \underbrace{\frac{60\,\kappa_r}{\chi\,\omega_{\mathrm{top}}}\Big(1+\frac1{\sqrt p}\Big)^2
> \Big(2\zeta^2+\frac{2(1-\beta)}{1+\beta}\sigma^2\Big)}_{\text{(III) correction 侧 Byzantine}}
> \tag{12.2}
> $$
> 两个定理同时成立，最终界取二者较小者。

### 12.3 关键结论：噪声 floor 可被 $\beta$ 任意压小

> **推论 12.3（$\beta$ 的作用）** (12.1)、(12.2) 的 error floor 可写为
> $$
> \mathcal F_{\mathrm{floor}}=\underbrace{\mathcal C_\zeta\,\zeta^2}_{\text{与 }\beta\text{ 无关}}
> +\underbrace{(1-\beta)\,\mathcal C_\sigma\,\sigma^2}_{\text{正比于 }1-\beta},
> $$
> $$
> \mathcal C_\zeta=O\Big(\frac{\kappa_q^\star}p+\frac{\kappa_r}{\omega_{\mathrm{top}}^2}\Big(1+\frac1{\sqrt p}\Big)^2\Big),
> \qquad
> \mathcal C_\sigma=O\Big(\frac{\kappa_q^\star}p+\frac1{Hc^2}+\frac{\kappa_r}{\omega_{\mathrm{top}}^2}\Big(1+\frac1{\sqrt p}\Big)^2\Big),
> $$
> 二者均与 $\beta$、$\eta$、$T$ 无关。

> **推论 12.4（联合取参：噪声 floor 完全消失）**
> 取
> $$
> \eta=\min\{\eta_{\max},\ \eta_0T^{-1/2}\},
> \qquad
> 1-\beta=\theta_0T^{-1/2}
> \tag{12.3}
> $$
> （$\eta_0,\theta_0>0$ 为与 $T$ 无关的常数，且满足 (11.18) 所要求的 $\eta_0\lesssim\theta_0$）。则
> $$
> \boxed{\
> \frac1T\sum_{t=0}^{T-1}\mathbb E\|\nabla f(\theta_t)\|^2
> \ =\ O\Big(\frac1{\sqrt T}\Big)
> \ +\ O\Big(\frac{\kappa_q^\star\,\zeta^2}p\Big)
> \ +\ O\Big(\frac{\kappa_r\,\zeta^2}{\omega_{\mathrm{top}}^2}\Big(1+\frac1{\sqrt p}\Big)^2\Big)\ }
> \tag{12.4}
> $$
> （末项只在定理 B 分支出现）。**error floor 中不再含 $\sigma^2$：全部采样噪声项以 $O(T^{-1/2})$ 的速率消失。**

*证明.* 由推论 12.3，噪声部分为 $(1-\beta)\mathcal C_\sigma\sigma^2=\theta_0\mathcal C_\sigma\sigma^2T^{-1/2}=O(T^{-1/2})$。优化项 $\frac{4\Delta_0}{\eta\lambda_AT}=O(T^{-1/2})$。$\Xi_0^\sharp/T$ 由 (11.20') 给出：其中含 $\frac1{1-\beta}$ 或 $\frac1{1-\beta^2}$ 的三项（第二、三、五项，第五项只在定理 B 分支出现，来自定理 7.6 的余项）除以 $T$ 后都是 $O\big(\frac1{(1-\beta)T}\big)=O(T^{-1/2})$；第一、四项除以 $T$ 后是 $O(1/T)$。相容性：(11.18) 要求 $\eta\lesssim(1-\beta)$，即 $\eta_0T^{-1/2}\lesssim\theta_0T^{-1/2}$，这是对常数的约束而非对 $T$ 的约束，故可满足。$\square$

> **推论 12.5（梯度复杂度）** 设 (12.4) 的 floor $\le\varepsilon^2/2$，则 $T=O(\varepsilon^{-4})$。

### 12.4 与变形A、与文献的逐项对照

| 项 | 变形A | 本版（客户端动量） |
|---|---|---|
| (I) tracker 侧 | $3\kappa_q^\star\Sigma^2/p$，$\Sigma^2=\sigma^2+\zeta^2$ | $6\kappa_q^\star\zeta^2/p+O((1-\beta)\sigma^2/p)$ |
| (II) 噪声 | $\frac{3\sigma^2}H+\frac{48\sigma^2}{Hc^2}$，**与 $\eta,\beta$ 无关，不消失** | 全部带 $(1-\beta)$ 或 $(1-\beta)^2$，**可任意压小** |
| (III) correction 侧 | $\frac{60\kappa_r\bar V_r}{\chi\omega_{\mathrm{top}}}$，$\bar V_r\propto\Sigma^2$ | 同形式但 $\bar V_r\propto\Sigma_\beta^2$，$\sigma$ 部分带 $(1-\beta)$ |
| clipping bias | 不存在（ARC + cap 只作用于二阶矩） | 同 |
| 离散化项 | 不存在（被 $\bar Z$ 吸收） | 同 |
| $\kappa_q^\star$ | $\le73$（$B=\hat B$ 时 $\le41$） | 同 |
| 步长条件 | $\eta\lesssim c\cdot\frac{\lambda_A}{L\Lambda_A^2}$ | 额外要求 $\eta\lesssim(1-\beta)$ |
| 最终 floor | $O\big(\frac{\Sigma^2}p+\frac{\sigma^2}{Hc^2}\big)$ | $O\big(\frac{\kappa_q^\star\zeta^2}p\big)$（**纯异质性**） |

> **注 12.6（与下界的距离）** (12.4) 的 floor 只含 $\zeta^2$，而 $(B,\zeta^2)$-异质性下任何 Byzantine-robust 算法都有下界 $\Omega\big(\frac BN\zeta^2\big)$（Karimireddy–He–Jaggi 2022；Allouah et al. 2023）。因此本版 floor 与下界的差距**只剩两个因子**：
> 1. **聚合器质量** $\kappa_q^\star=O(1)$ 对下界的 $\frac BN$——用 NNM 预聚合可降到 $O(\hat B/N)$，**与下界匹配**（§14.7）；
> 2. **压缩代价** $1/p$——这是本工作留下的主要理论问题（§16 第 1 条）。
>
> 换言之，**在客户端动量之后，本文的 error floor 与已知下界的全部差距就是那一个压缩代价因子**。这是相对变形A（其 floor 还含不可消除的 $\sigma^2/(Hc^2)$）的实质改进。

> **注 12.7（客户端动量的文献依据）** 用客户端动量压小诚实方 dispersion 从而改善 Byzantine 鲁棒性，是 Karimireddy–He–Jaggi (ICML 2021) 提出的机制（本文参考文献第 2 篇，此前只引用了其 $(f,\kappa)$-robustness 部分）。Byz-EF21-SGDM (TNNLS'26) 与 Byz-DM21 (AISTATS'26) 均采用同一路线。本版的新意不在动量本身，而在于：
> * 动量与 **EF21 tracker + 全局一致 mask** 的相容性——(7.5)/(7.7) 的正交分解结构在把 $g$ 换成 $u$ 后**一字未变**（注 7.7）；
> * 动量与 **cap 只作用于二阶矩**的相容性——两者产生的两个 $\eta^2\bar Z$ 系数在同一个自洽闭合中一并吸收（定理 11.15 Step 3）。

---

## 13. 参数选取

### 13.1 各参数的作用汇总

| 参数 | 出现处 | 增大的效果 |
|---|---|---|
| $p=\frac{k_{\mathrm{rnd}}}{m-k_{\mathrm{top}}}$ | (7.6)(7.8)(9.3)(12.1)(12.2)(I)(II) | tracker/correction dispersion 减小、$\chi$ 增大；但挤占 $k_{\mathrm{top}}$ |
| $\omega_{\mathrm{top}}$ | (9.11)(12.2)(III) | tracking 效率提高；但需要更大的 $k_{\mathrm{top}}$ |
| $\kappa_q^\star$ | (8.2)(10.6)(12.1)(12.2)(I) | error floor 主项线性变差；含 ARC 的 $O(1)$ 放大 |
| $\kappa_r$ | (8.1)(9.11)(12.2)(III) | 仅影响定理 B 的 (III) 项 |
| $r$ | $\delta_{\mathrm{JL}},\varepsilon_{\mathrm{JL}}$（§14.1） | JL 畸变减小、$\omega_{\mathrm{top}}$ 增大；通信量 $\frac rn$ 增大 |
| $\beta$（客户端动量） | (7.4)(11.6)(11.11)(12.1) | **增大 $\beta$ 使 error floor 中全部 $\sigma^2$ 项按 $(1-\beta)$ 缩小**（推论 12.3）；代价是步长上限按 $(1-\beta)$ 收紧（注 11.16）。推荐 $1-\beta=\Theta(T^{-1/2})$（推论 12.4） |
| $C_{\max}$ | 仅经 $\lambda_A$ 进入 $C_A^\sharp$ | **单调有害**：增大只使公共系数以 $C_{\max}^2$ 增长，不再有任何抵消项（§13.2） |
| $k_{\mathrm{arc}}$ | — | **不是自由参数**，由 (2.0) 确定 |

### 13.2 $C_{\max}$ 的取舍

本版中 $C_{\max}$ **只经 $\lambda_A=1/(C_{\max}+\sqrt\epsilon)$ 进入公共系数**：

$$
C_A^\sharp=\frac{4\Lambda_A^2\big(1+6c_\beta\big)}{\lambda_A^2}+1
=\frac{4\big(1+6c_\beta\big)\big(C_{\max}+\sqrt\epsilon\big)^2}{\epsilon}+1 .
\tag{13.1}
$$

与完整版的关键差别：**(13.1) 关于 $C_{\max}$ 单调递增，而不再有任何抵消项**。完整版中增大 $C_{\max}$ 可以减小 $\mathcal B_{\mathrm{clip}}^{(T)}$，二者构成一个有内点最优解的权衡；本版 $\mathcal B_{\mathrm{clip}}^{(T)}$ 已不存在，因此**理论上 $C_{\max}$ 越小越好**。

这需要如实说明其含义与边界：

> **注 13.1（$C_{\max}\to0$ 的退化与理论的局限）**
> $C_{\max}\to0$ 时 $\hat g_t\to0$，故 $v_{t+1}\to0$，preconditioner 退化为常数 $1/\sqrt\epsilon$，算法变成**带动量的 SGD**：
> $\theta_{t+1}=\theta_t-\tfrac\eta{\sqrt\epsilon}m_{t+1}$。此时 $\lambda_A=\Lambda_A$、$C_A^\sharp=8(1+6c_\beta)+1$ 取到最小值。
> 换言之，(13.1) 的单调性反映的不是"应当取 $C_{\max}\to0$"，而是**本文的界（与文献中所有最坏情况 Adam 分析一样）看不到自适应性带来的好处**：
> 它对 SGDM 给出的常数总是不劣于对 Adam 给出的常数。因此 $C_{\max}$ 应由**实践**而非本界决定，
> 其角色是"保留多少自适应性"的旋钮，而理论为此收取 $(C_{\max}+\sqrt\epsilon)^2/\epsilon$ 的常数。

> **注 13.2（实践上如何取 $C_{\max}$，以及一个可监测判据）**
> 本版把裁剪从更新路径移走之后，出现一个**新的实践约束**：若 $\|z_t\|$ 频繁超过 $C_{\max}$，则 $v$ 系统性地低估二阶矩，
> 有效步长偏大——这与"裁剪用于稳定训练"的常规直觉相反。
> 稳定性在本版中由 **ARC 承担**（它在聚合前已把每个 tracker 压到诚实方水平），$C_{\max}$ 不再负责稳定性；
> 但仍应取得足够大，使 $v$ 是二阶矩的合理估计。
> 引理 10.8 给出一个可直接监测的判据：$\|z_t\|>C_{\max}$ 要求
> $$
> \max_{i\in\mathcal H}\big\|q_t^{(i)}\big\|>C_{\max},
> $$
> 即**诚实方 tracker 范数最大值**超过 $C_{\max}$——这是一个纯诚实量的尾部事件，与 Byzantine 行为无关，可在训练中直接统计其触发频率。
> 建议取 $C_{\max}$ 为该最大值经验分布的一个高分位数，使触发频率保持在很低的水平。

> **注 13.3（无 cap 不可行）** $C_{\max}=\infty$ 时 $v_{t+1}$ 无上界，$\lambda_A=0$，引理 11.13 的下降项消失，整个分析崩塌。
> 文献中去掉 cap 的两条路都要付出别的代价：假设梯度有界（Reddi et al. 2021、Défossez et al. 2022），
> 或把噪声假设加强到几乎必然有界/次高斯并退化为高概率结论（Li–Rakhlin–Jadbabaie 2023、Hong–Lin 2023）。
> 本文选择第三条路——在算法中构造性地实现它——这是在只用 A3（期望型有界方差）的前提下保持期望型平方范数结论的唯一已知做法。

### 13.3 refresh 比例 $\rho$ 的最优选取

令总行压缩比 $\gamma=k/m$、refresh 占比 $\rho=k_{\mathrm{rnd}}/k$，即 $k_{\mathrm{rnd}}=\rho k$、$k_{\mathrm{top}}=(1-\rho)k$。则

$$
p=\frac{\rho\gamma}{1-(1-\rho)\gamma},
\qquad
\omega_{\mathrm{top}}=\theta_{\mathrm{JL}}^{-1}(1-\rho)\gamma
\quad(\text{见 (14.21)}),
\tag{13.3}
$$

其中 $\theta_{\mathrm{JL}}=\frac{1+\varepsilon_{\mathrm{JL}}}{1-\varepsilon_{\mathrm{JL}}}$。二者对 $\rho$ 单调相反，构成真实的权衡。

在 $\gamma\ll1$ 的压缩区间内，$p\approx\rho\gamma$，故 (12.3) 的与 $T$ 无关部分主要由

$$
\Psi(\rho)
:=\frac{\kappa_q^\star\Sigma_\beta^2}{\rho\gamma}
+\frac{\kappa_r\Sigma_\beta^2\,\theta_{\mathrm{JL}}^2}{(1-\rho)^2\gamma^2}\Big(1+\frac1{\sqrt{\rho\gamma}}\Big)^2
\tag{13.4}
$$

主导。当 $\rho\gamma\ll1$ 时 $(1+(\rho\gamma)^{-1/2})^2\approx\frac1{\rho\gamma}$，于是

$$
\Psi(\rho)\approx\frac{\Sigma_\beta^2}\gamma\bigg[\frac{\kappa_q^\star}\rho+\frac{\kappa_r\theta_{\mathrm{JL}}^2}{\gamma^2\rho(1-\rho)^2}\bigg]
=\frac{\Sigma_\beta^2}{\gamma\rho}\bigg[\kappa_q^\star+\frac{\kappa_r\theta_{\mathrm{JL}}^2}{\gamma^2(1-\rho)^2}\bigg].
$$

> **推论 13.1（$\rho^\star$）** 记 $\varkappa:=\dfrac{\kappa_r\theta_{\mathrm{JL}}^2}{\kappa_q^\star\gamma^2}$。则 $\Psi$ 的（近似）最小点 $\rho^\star$ 是方程
> $$
> \boxed{\ (1-\rho)^3=\varkappa\,(3\rho-1)\ }
> \tag{13.5}
> $$
> 在区间 $\big(\tfrac13,1\big)$ 内的**唯一**根，且
> $$
> \varkappa\to0\ (\text{correction 侧误差可忽略}):\quad\rho^\star\to1^- ,
> \qquad
> \varkappa\to\infty\ (\text{correction 侧误差主导}):\quad\rho^\star\to\tfrac13^+ .
> $$
> 特别地 $\rho^\star>\tfrac13$ 恒成立，即 **random refresh 至少应占总行预算的三分之一**。

*推导.* 令 $\phi(\rho)=\frac1\rho\big[\kappa_q+\kappa_r\theta_{\mathrm{JL}}^2\gamma^{-2}(1-\rho)^{-2}\big]$。则

$$
\phi'(\rho)=-\frac1{\rho^2}\Big[\kappa_q+\frac{\kappa_r\theta_{\mathrm{JL}}^2}{\gamma^2(1-\rho)^2}\Big]
+\frac1\rho\cdot\frac{2\kappa_r\theta_{\mathrm{JL}}^2}{\gamma^2(1-\rho)^3},
$$

令 $\phi'(\rho)=0$，两边乘 $\rho^2(1-\rho)^3/\kappa_q$ 并以 $\varkappa$ 记比值：

$$
(1-\rho)^3+\varkappa(1-\rho)=2\varkappa\rho
\quad\Longleftrightarrow\quad
(1-\rho)^3=\varkappa(3\rho-1),
$$

即 (13.5)。左端在 $\rho\in(\frac13,1)$ 上由 $\frac8{27}$ 单调降至 $0$，右端由 $0$ 单调升至 $2\varkappa$，故在 $(\frac13,1)$ 内恰有一根；$\rho\le\frac13$ 时右端 $\le0<$ 左端，无根，故 $\phi'<0$，$\phi$ 在 $(0,\frac13]$ 上递减。极限行为由 (13.5) 直接读出。$\square$

**实践建议**：$\gamma$ 通常较小（强压缩），故 $\varkappa=\kappa_r\theta_{\mathrm{JL}}^2/(\kappa_q\gamma^2)$ 往往很大，此时 $\rho^\star\approx\frac13$，即 **约三分之一的行预算用于 random refresh、三分之二用于 Top-K**。这与"refresh 只需保证 tracker 不永久 stale、Top-K 承担信息选择"的直观一致。若只关心定理 A（不使用投影侧保证），则 (12.1) 中 $p$ 越大越好，应取 $\rho\to1$。

### 13.4 投影维数与通信量

由推论 14.2 与定理 12.2 的前提 $\delta_{\mathrm{JL}}\le\omega_{\mathrm{top}}/4$，取

$$
\varepsilon_{\mathrm{JL}}=\tfrac13
\ (\Rightarrow\theta_{\mathrm{JL}}=2),
\qquad
\delta_{\mathrm{JL}}=\frac{\omega_{\mathrm{top}}}4,
\qquad
r=\Big\lceil 9c_0\Big(\min\{N,n\}+\log\frac{4m}{\omega_{\mathrm{top}}}\Big)\Big\rceil ,
\tag{13.6}
$$

则 $\omega_{\mathrm{top}}=\frac12(1-\rho)\gamma$，总上行压缩比

$$
\rho_{\mathrm{comm}}
=\frac rn+\gamma
=O\bigg(\frac{\min\{N,n\}+\log\frac m\gamma}n\bigg)+\gamma .
\tag{13.7}
$$

在联邦学习的典型区间 $N\ll n$（客户端数远小于 reshape 后的列数）内，第一项为 $O\big(\frac{N+\log(m/\gamma)}n\big)=o(1)$，故投影阶段的通信开销可忽略，总压缩比由 $\gamma=k/m$ 主导。**注意 $r$ 与 $T$ 无关**（引理 9.3 消除了对 $T$ 的 union bound），这是本文相对原草稿方案的一个实质改进。

---

# 第 II 部分：抽象假设的可实现性

第 I 部分的全部结论建立在 A5（$(f,\kappa_r,s)$-robust subset）、A6（$(f,\kappa_q)$-robust aggregation）、A7（$(\omega_{\mathrm{top}},\delta_{\mathrm{JL}})$-projected Top-K contraction）之上。本部分证明这三条假设**不是空假设**：Algorithm 1 中把 $\mathcal R_r$ 与 $\mathcal A_q$ 都取为 **Multi-Krum**，配合单套 Gaussian 投影，可以**同时**满足 A5、A6、A7，并给出全部常数的显式表达式。

本部分的逻辑链条是

$$
\underbrace{\text{subspace embedding}}_{\S14.1}
\ \Longrightarrow\
\begin{cases}
\underbrace{\text{成对距离保持}}_{(14.6)}\Longrightarrow\underbrace{\text{投影 Multi-Krum 的近似最小性}}_{\S14.3}\Longrightarrow\ \text{A5}\ (\S14.4)\\[4pt]
\underbrace{\text{子集平均的 row energy 保持}}_{(14.7)}\Longrightarrow\ \text{A7}\ (\S14.5)
\end{cases}
$$

而 A6 由全维 Multi-Krum 直接给出（§14.6），不涉及投影。

> **为什么 Multi-Krum 是投影空间的自然选择。** 两个结构性事实使它与本算法的第一阶段完美契合：
> 1. **Multi-Krum 的打分只依赖成对距离**，而 Johnson–Lindenstrauss 型投影恰好保持成对距离。因此"在投影空间打分"与"在原空间打分"只相差一个 $(1\pm\varepsilon_{\mathrm{JL}})$ 因子。
> 2. **Multi-Krum 的输出本身就是"某个大小 $s$ 的子集的平均"**，与 A5 所要求的 $\bar r_t^S=\frac1s\sum_{i\in\mathcal S_t}r_t^{(i)}$ 形式完全一致；服务器只需回传下标集 $\mathcal S_t$，无需在投影空间重构任何全维向量。
>
> 相比之下，几何中位数、coordinate-wise median、CWTM 等规则的输出不是输入的子集平均（前者非线性，后两者逐坐标混合），在投影空间计算后无法映射回原空间的对应操作。§14.7 讨论它们的可用方式。

## 14.1 统一的 subspace-embedding good event

对第 $t$ 轮与每个行索引 $j\in[m]$，把 $N$ 个客户端该行的 correction 堆叠：

$$
A_{t,j}:=
\begin{bmatrix}
(R_t^{(1)})_{j,:}\\ \vdots\\ (R_t^{(N)})_{j,:}
\end{bmatrix}
\in\mathbb R^{N\times n},
\qquad
\mathrm{rank}(A_{t,j})\le d_\star:=\min\{N,n\}.
\tag{14.1}
$$

由 Condition 1 与 (M1)，$A_{t,j}$ 是 $\mathcal F_t^{c}$-可测的（**包括 Byzantine 行**），因而独立于 $V_t$。

> **引理 14.1（Oblivious subspace embedding）** 设 $A\in\mathbb R^{N\times n}$ 满足 $\mathrm{rank}(A)\le d_\star$，$V\in\mathbb R^{n\times r}$ 元素 i.i.d. $\mathcal N(0,1)$，$\Pi:=V/\sqrt r$。存在绝对常数 $c_0$，使当
> $$
> r\ \ge\ c_0\,\varepsilon^{-2}\Big(d_\star+\log\tfrac1{\delta'}\Big)
> \tag{14.2}
> $$
> 时，以至少 $1-\delta'$ 的概率
> $$
> (1-\varepsilon)\big\|y^\top A\big\|^2
> \le\big\|y^\top A\Pi\big\|^2
> \le(1+\varepsilon)\big\|y^\top A\big\|^2,
> \qquad\forall y\in\mathbb R^N .
> \tag{14.3}
> $$

*说明.* 这是 Gaussian 随机矩阵的标准 subspace embedding 性质：把 $\mathrm{rowspace}(A)$（维数 $\le d_\star$）的一组正交基取出，(14.3) 等价于 $\Pi$ 在该子空间上的奇异值全部落在 $[\sqrt{1-\varepsilon},\sqrt{1+\varepsilon}]$，由 Gaussian 矩阵奇异值的集中不等式（如 Vershynin 2018, Thm 4.6.1）与 net 论证即得。参见 Sarlós (2006)、Woodruff (2014, Thm 2.3)。

**定义（good event）**

$$
\mathcal E_t:=
\bigcap_{j=1}^m
\Big\{
(1-\varepsilon_{\mathrm{JL}})\|y^\top A_{t,j}\|^2
\le\big\|y^\top A_{t,j}\tfrac{V_t}{\sqrt r}\big\|^2
\le(1+\varepsilon_{\mathrm{JL}})\|y^\top A_{t,j}\|^2,
\ \forall y\in\mathbb R^N
\Big\}.
\tag{14.4}
$$

$\mathcal E_t\in\mathcal G_t$（它是 $V_t$ 与 $\mathcal F_t^{c}$ 的函数）。

> **推论 14.2（投影维数）** 若
> $$
> \boxed{\
> r\ \ge\ c_0\,\varepsilon_{\mathrm{JL}}^{-2}
> \Big(\min\{N,n\}+\log\tfrac m{\delta_{\mathrm{JL}}}\Big)\ }
> \tag{14.5}
> $$
> 则 $\Pr\big(\mathcal E_t\mid\mathcal F_t^{c}\big)\ge1-\delta_{\mathrm{JL}}$，对每个 $t$ 一致成立。即 **A7 的 (3.6) 成立**。

*证明.* 对每个 $j$ 用引理 14.1（$\delta'=\delta_{\mathrm{JL}}/m$）并对 $j\in[m]$ 取 union bound；因 $A_{t,j}$ 独立于 $V_t$，该概率是条件于 $\mathcal F_t^{c}$ 的。$\square$

**(14.5) 相对原草稿的两点改进。** 原草稿为处理 $\mathcal S_t=\mathcal S_t(V_t)$ 与同一投影的依赖，对全部 $\binom Ns$ 个候选子集做同时保持，得

$$
r\gtrsim\varepsilon_{\mathrm{JL}}^{-2}\log\frac{mT\binom Ns}{\delta}
\qquad(\text{原草稿问题 §3、§8}).
$$

1. **$\log\binom Ns$ 被 $\min\{N,n\}$ 取代。** 二者在 $s=\Theta(N)$ 时同阶（$\log\binom Ns=\Theta(N)$），但 (14.5) 的形式不含 $s$、常数干净，且当 $n<N$ 时自动改善为 $n$。
2. **$\log T$ 消失。** 由引理 9.3，只需**每轮**失败概率 $\le\omega_{\mathrm{top}}/4$，无需对 $T$ 轮做 union bound。

更重要的是 (14.4) 是对**全体** $y\in\mathbb R^N$ 的一致保证，因此**一个事件同时给出 A5 与 A7 所需的两条性质**：

> **推论 14.3（在 $\mathcal E_t$ 上）**
> **(a) 成对距离保持**（取 $y=e_i-e_{i'}$，对 $j\in[m]$ 求和）：$\forall i,i'\in[N]$，
> $$
> (1-\varepsilon_{\mathrm{JL}})\big\|r_t^{(i)}-r_t^{(i')}\big\|^2
> \le\big\|P_t^{(i)}-P_t^{(i')}\big\|_F^2
> \le(1+\varepsilon_{\mathrm{JL}})\big\|r_t^{(i)}-r_t^{(i')}\big\|^2 .
> \tag{14.6}
> $$
> **(b) 任意子集平均的逐行能量保持**（取 $y=\mathbf 1_S/|S|$）：$\forall S\subseteq[N]$，$\forall j\in[m]$，
> $$
> (1-\varepsilon_{\mathrm{JL}})\big\|(\bar R_t^{S})_{j,:}\big\|^2
> \le\big\|(\bar P_t^{S})_{j,:}\big\|^2
> \le(1+\varepsilon_{\mathrm{JL}})\big\|(\bar R_t^{S})_{j,:}\big\|^2 .
> \tag{14.7}
> $$

记 $\theta_{\mathrm{JL}}:=\dfrac{1+\varepsilon_{\mathrm{JL}}}{1-\varepsilon_{\mathrm{JL}}}\ge1$。

## 14.2 Multi-Krum 与其全空间鲁棒性

> **定义 14.4（Multi-Krum）** 输入 $x_1,\ldots,x_N\in\mathbb R^D$，参数：已知 Byzantine 上界 $\hat B$、输出规模 $s$。令邻居数
> $$
> \nu:=N-\hat B-2 .
> \tag{14.8}
> $$
> 对每个 $i\in[N]$ 定义 **Krum score**
> $$
> \mathrm{sc}(i):=
> \min_{\substack{\mathcal N\subseteq[N]\setminus\{i\}\\|\mathcal N|=\nu}}
> \ \sum_{j\in\mathcal N}\|x_i-x_j\|^2
> \tag{14.9}
> $$
> （即取 $i$ 的 $\nu$ 个最近邻的平方距离之和）。令 $\mathcal S$ 为 score 最小的 $s$ 个下标（并列时按下标次序定），输出
> $$
> \mathrm{MK}_s(x_1,\ldots,x_N):=\bar x_{\mathcal S}=\frac1s\sum_{i\in\mathcal S}x_i .
> \tag{14.10a}
> $$

> **命题 14.5（Multi-Krum 的 $(f,\kappa,s)$-robustness，任意参照子集版）**
> 设 $\nu=N-\hat B-2$、$\nu_\star:=N-2\hat B-2$，并设 A8（$N\ge2\hat B+3$，故 $\nu_\star\ge1$）。
> 设 $\mathcal G\subseteq[N]$ 为**任意**满足
> $$
> g:=|\mathcal G|\ \ge\ N-\hat B
> \tag{14.10}
> $$
> 的下标集（**不必是诚实集**；条件 (14.10a)），记
> $$
> \bar x_{\mathcal G}=\tfrac1g\!\!\sum_{i\in\mathcal G}\!x_i,
> \qquad
> V_{\mathcal G}:=\tfrac1g\!\!\sum_{i\in\mathcal G}\!\|x_i-\bar x_{\mathcal G}\|^2 .
> $$
> 若 $s\le g$，则对任意输入 $x_1,\ldots,x_N$（$\mathcal G$ 之外的分量任意）**确定性地**
> $$
> \big\|\bar x_{\mathcal S}-\bar x_{\mathcal G}\big\|^2
> \ \le\ \kappa(g)\,V_{\mathcal G},
> \qquad
> \kappa(g)=\frac{2g}{\nu_\star}\bigg(\frac{2\nu g}{(g-1)(g-s+1)}+1\bigg).
> \tag{14.11}
> $$

> **推论 14.5a（两个用法）**
> **(a) 诚实集版（用于 A5）**：取 $\mathcal G=\mathcal H$（$g=H=N-B\ge N-\hat B$，(14.10a) 成立），得原版的 (14.11)，$\kappa=\kappa(H)$。取 $s=\lceil H/2\rceil$、$\hat B\le N/4$、$N\ge8$ 时 $\kappa(H)\le72$。
> **(b) 标准 $(\hat B,\kappa)$-robustness 版（用于定理 10.4）**：取 $g=N-\hat B$，则 (14.11) **正是定义 10.5**（其中的 $\mathcal G$ 即定义 10.5 的 $S$）。此时 $\nu=g-2$，故
> $$
> \kappa_q:=\kappa(N-\hat B)=\frac{2g}{\nu_\star}\bigg(\frac{2(g-2)g}{(g-1)(g-s_q+1)}+1\bigg),
> \qquad g=N-\hat B .
> \tag{14.11a}
> $$
> 取 $s_q=\lceil g/2\rceil$、$\hat B\le N/4$、$N\ge8$ 时
> $$
> \kappa_q\ \le\ 40 .
> \tag{14.12}
> $$

*证明（命题 14.5）.* 分四步。记 $\mathrm{sc}$ 为 (14.9)。全程只用到 $\mathcal G$ 的**大小**与 $|[N]\setminus\mathcal G|\le\hat B$，不用到 $\mathcal G$ 的任何其他性质——这正是本版相对原版的推广所在。

**Step 1（$\mathcal G$ 内 score 的上界）.** 固定 $i\in\mathcal G$。由 (14.10a)，$|\mathcal G\setminus\{i\}|=g-1\ge N-\hat B-1>\nu$，故 (14.9) 的可行域包含 $\mathcal G\setminus\{i\}$ 的全部 $\nu$-子集，于是 $\mathrm{sc}(i)$ 不超过其中任一个，进而不超过它们的平均：

$$
\mathrm{sc}(i)\le\frac\nu{g-1}\sum_{j\in\mathcal G\setminus\{i\}}\|x_i-x_j\|^2
=\frac\nu{g-1}\sum_{j\in\mathcal G}\|x_i-x_j\|^2 .
$$

由 $\sum_{j\in\mathcal G}(x_j-\bar x_{\mathcal G})=0$，

$$
\sum_{j\in\mathcal G}\|x_i-x_j\|^2
=\sum_{j\in\mathcal G}\big\|(x_i-\bar x_{\mathcal G})-(x_j-\bar x_{\mathcal G})\big\|^2
=g\|x_i-\bar x_{\mathcal G}\|^2+gV_{\mathcal G} .
\tag{14.13}
$$

故记 $A_{\mathcal G}:=\dfrac{\nu g}{g-1}$，有 $\mathrm{sc}(i)\le A_{\mathcal G}\big(\|x_i-\bar x_{\mathcal G}\|^2+V_{\mathcal G}\big)$，$\forall i\in\mathcal G$。

**Step 2（被选中者的 score 上界）.** 对 $\mathcal G$ 求和并用 $\sum_{i\in\mathcal G}\|x_i-\bar x_{\mathcal G}\|^2=gV_{\mathcal G}$：

$$
\sum_{i\in\mathcal G}\mathrm{sc}(i)\le2A_{\mathcal G}\,gV_{\mathcal G} .
$$

把 $\mathcal G$ 内的 score 升序排列为 $\mathrm{sc}^{\mathcal G}_{(1)}\le\cdots\le\mathrm{sc}^{\mathcal G}_{(g)}$。因其中有 $g-s+1$ 个不小于 $\mathrm{sc}^{\mathcal G}_{(s)}$（此处用 $s\le g$），

$$
(g-s+1)\,\mathrm{sc}^{\mathcal G}_{(s)}\le\sum_{i\in\mathcal G}\mathrm{sc}(i)\le2A_{\mathcal G}gV_{\mathcal G}
\quad\Longrightarrow\quad
\mathrm{sc}^{\mathcal G}_{(s)}\le\frac{2A_{\mathcal G}gV_{\mathcal G}}{g-s+1}=:\mathcal M_{\mathcal G} .
\tag{14.14}
$$

另一方面，有 $s$ 个属于 $\mathcal G$ 的下标其 score $\le\mathrm{sc}^{\mathcal G}_{(s)}$，故**全体** $N$ 个 score 中第 $s$ 小者 $\le\mathrm{sc}^{\mathcal G}_{(s)}$。由 $\mathcal S$ 是 score 最小的 $s$ 个，

$$
\mathrm{sc}(k)\ \le\ \mathrm{sc}^{\mathcal G}_{(s)}\ \le\ \mathcal M_{\mathcal G},
\qquad\forall k\in\mathcal S .
\tag{14.15}
$$

**Step 3（被选中者靠近 $\bar x_{\mathcal G}$）.** 固定 $k\in\mathcal S$，设 $\mathcal N_k$ 为 (14.9) 取到最小值的邻居集，$|\mathcal N_k|=\nu$。其中落在 $\mathcal G$ 之外的至多 $|[N]\setminus\mathcal G|=N-g\le\hat B$ 个（用 (14.10a)），故

$$
\big|\mathcal N_k\cap\mathcal G\big|\ \ge\ \nu-\hat B=N-2\hat B-2=\nu_\star\ \ge1 .
$$

由 $\sum_{j\in\mathcal N_k\cap\mathcal G}\|x_k-x_j\|^2\le\mathrm{sc}(k)\le\mathcal M_{\mathcal G}$ 与 $\sum_{j\in\mathcal N_k\cap\mathcal G}\|x_j-\bar x_{\mathcal G}\|^2\le\sum_{i\in\mathcal G}\|x_i-\bar x_{\mathcal G}\|^2=gV_{\mathcal G}$，对 $\mathcal N_k\cap\mathcal G$ 取平均并用 $\|u+v\|^2\le2\|u\|^2+2\|v\|^2$：

$$
\|x_k-\bar x_{\mathcal G}\|^2
\le\frac1{|\mathcal N_k\cap\mathcal G|}\sum_{j\in\mathcal N_k\cap\mathcal G}
\Big(2\|x_k-x_j\|^2+2\|x_j-\bar x_{\mathcal G}\|^2\Big)
\le\frac2{\nu_\star}\big(\mathcal M_{\mathcal G}+gV_{\mathcal G}\big).
$$

**Step 4（子集平均）.** 由 (J)，

$$
\big\|\bar x_{\mathcal S}-\bar x_{\mathcal G}\big\|^2
=\Big\|\frac1s\sum_{k\in\mathcal S}(x_k-\bar x_{\mathcal G})\Big\|^2
\le\frac1s\sum_{k\in\mathcal S}\|x_k-\bar x_{\mathcal G}\|^2
\le\frac2{\nu_\star}\big(\mathcal M_{\mathcal G}+gV_{\mathcal G}\big),
$$

代入 (14.14) 的 $\mathcal M_{\mathcal G}$ 与 $A_{\mathcal G}=\frac{\nu g}{g-1}$ 即得 (14.11)。$\square$

*证明（推论 14.5a）.* **(a)** $\mathcal G=\mathcal H$ 时 $g=H=N-B\ge N-\hat B$，(14.10) 成立；$|[N]\setminus\mathcal H|=B\le\hat B$，Step 3 的计数成立。数值：取 $s=\lceil H/2\rceil$ 则 $H-s+1\ge H/2$，故 $\frac{2\nu H}{(H-1)(H-s+1)}\le\frac{4\nu}{H-1}$；在 $\hat B\le N/4$、$N\ge8$ 下 $H\ge\frac{3N}4$、$\nu\le N$、$\nu_\star\ge\frac N2-2\ge\frac N4$、$H-1\ge\frac N2$，于是 $\frac{4\nu}{H-1}\le8$、$\frac{2H}{\nu_\star}\le8$，得 $\kappa(H)\le8\times9=72$。

**(b)** $g=N-\hat B$ 时 (14.10) 取等号，(14.11) 的陈述与定义 10.5 逐字相同（把 $\mathcal G$ 读作 $S$、$V_{\mathcal G}$ 读作 $\frac1{|S|}\sum_{i\in S}\|x_i-\bar x_S\|^2$）。数值上此时 $\nu=N-\hat B-2=g-2$，故

$$
\frac{2\nu g}{(g-1)(g-s_q+1)}=\frac{2(g-2)g}{(g-1)(g-s_q+1)}
\ \overset{s_q=\lceil g/2\rceil}{\le}\ \frac{4(g-2)}{g-1}\ <\ 4 ,
$$

而 $\hat B\le N/4$、$N\ge8$ 给出 $g=N-\hat B\le N$、$\nu_\star\ge\frac N4$，故 $\frac{2g}{\nu_\star}\le8$。合起来 $\kappa_q\le8\times(4+1)=40$。$\square$

> **注 14.5b（这个推广解决了什么，以及为什么常数反而变好）**
> 原版命题 14.5 只对诚实集 $\mathcal H$ 陈述，而定理 10.4（ARC 保持鲁棒性）建立在**标准**定义 10.5 之上，后者对**所有**大小 $N-\hat B$ 的子集量化。$B<\hat B$ 时 $\mathcal H$ 比该尺寸大，因此不是定义 10.5 中的容许子集，定理 10.4 无法直接引用——这是变形A 初稿中标注的缺口。
> 上面的推广把证明中「诚实集」这一身份完全去掉：四步论证只用到 $|\mathcal G|=g$ 与 $|[N]\setminus\mathcal G|\le\hat B$ 两件事。取 $g=N-\hat B$ 即得定义 10.5，缺口消失，**引理 10.6 的归约不再需要**。
> 常数反而从 $\kappa_q^\star\le579$（初稿经引理 10.6 归约的结果）改善到 $\le73$，在 $B=\hat B$ 的最坏情形下更收紧到 $\le41$（推论 10.7），原因有二：一是省掉了引理 10.6 那两次三角不等式带来的因子 $2$；二是 $g=N-\hat B$ 时恰有 $\nu=g-2$，使 (14.11) 括号内的项从原版的界 $8$ 收紧到 $<4$。

> **注 14.6（与文献的关系）** 命题 14.5 是 Krum 的 $(f,\kappa)$-robustness（Blanchard et al. 2017 的原始论证；Allouah et al. 2023 给出 Krum 的 $\kappa=6(1+\frac f{n-2f})$）向 **Multi-Krum（$s>1$）** 与"子集平均"输出形式的推广。$\kappa=O(1)$ 与 Krum 同量级——Krum 族的 $\kappa$ 不随 $B/N\to0$ 衰减，这是该族的已知局限；§14.7 说明用 NNM 预聚合可改善到 $O(B/N)$，从而匹配下界。
> 另需注意 $s$ 的取值权衡：(14.11) 中 $\kappa$ 关于 $s$ 单调递增（经 $\frac1{H-s+1}$），故 $s$ 越小 $\kappa$ 越好；但 $s$ 太小会使子集平均的方差缩减效果变差（$\bar r_t^S$ 的随机噪声约为 $\sigma^2/s$，虽然本文的界未利用这一点）。$s=\lceil H/2\rceil$ 是使 $\kappa=O(1)$ 的一个方便取法；$s=O(1)$ 时 $\kappa$ 可进一步减小到 $\frac{2H}{\nu_H}(\frac{2\nu H}{(H-1)^2}+1)=O(1)$ 的更小常数。

## 14.3 投影不破坏 Multi-Krum：近似最小性

算法第 5 步在**投影空间**运行 Multi-Krum，即用

$$
\mathrm{sc}^{\mathrm{proj}}(i)=
\min_{\substack{\mathcal N\subseteq[N]\setminus\{i\}\\|\mathcal N|=\nu}}
\ \sum_{j\in\mathcal N}\big\|P_t^{(i)}-P_t^{(j)}\big\|_F^2
\tag{14.16}
$$

代替全空间 score $\mathrm{sc}^{\mathrm{full}}(i)$（后者定义在 $\{r_t^{(i)}\}$ 上）。下面说明这只带来一个 $\theta_{\mathrm{JL}}$ 因子。

> **引理 14.7（score 的双侧保持与近似最小性）** 在 $\mathcal E_t$ 上：
> **(a)** 对每个 $i\in[N]$，
> $$
> (1-\varepsilon_{\mathrm{JL}})\,\mathrm{sc}^{\mathrm{full}}(i)
> \ \le\ \mathrm{sc}^{\mathrm{proj}}(i)
> \ \le\ (1+\varepsilon_{\mathrm{JL}})\,\mathrm{sc}^{\mathrm{full}}(i).
> \tag{14.17}
> $$
> **(b)** 设 $\mathcal S_t$ 为 $\mathrm{sc}^{\mathrm{proj}}$ 最小的 $s$ 个下标。则
> $$
> \mathrm{sc}^{\mathrm{full}}(k)\ \le\ \theta_{\mathrm{JL}}\cdot\mathrm{sc}^{\mathrm{full},\mathcal H}_{(s)},
> \qquad\forall k\in\mathcal S_t ,
> \tag{14.18}
> $$
> 其中 $\mathrm{sc}^{\mathrm{full},\mathcal H}_{(s)}$ 是 honest 全空间 score 的第 $s$ 小值。

*证明.* **(a)** 对**固定**的邻居集 $\mathcal N$ 记 $F(\mathcal N)=\sum_{j\in\mathcal N}\|r_t^{(i)}-r_t^{(j)}\|^2$ 与 $F^{\mathrm{proj}}(\mathcal N)=\sum_{j\in\mathcal N}\|P_t^{(i)}-P_t^{(j)}\|_F^2$。由 (14.6) 逐项有 $(1-\varepsilon_{\mathrm{JL}})F(\mathcal N)\le F^{\mathrm{proj}}(\mathcal N)\le(1+\varepsilon_{\mathrm{JL}})F(\mathcal N)$。设 $\mathcal N^\ast=\arg\min F$、$\mathcal N'=\arg\min F^{\mathrm{proj}}$，则

$$
\mathrm{sc}^{\mathrm{proj}}(i)=F^{\mathrm{proj}}(\mathcal N')\le F^{\mathrm{proj}}(\mathcal N^\ast)\le(1+\varepsilon_{\mathrm{JL}})F(\mathcal N^\ast)=(1+\varepsilon_{\mathrm{JL}})\mathrm{sc}^{\mathrm{full}}(i),
$$
$$
\mathrm{sc}^{\mathrm{proj}}(i)=F^{\mathrm{proj}}(\mathcal N')\ge(1-\varepsilon_{\mathrm{JL}})F(\mathcal N')\ge(1-\varepsilon_{\mathrm{JL}})\mathrm{sc}^{\mathrm{full}}(i).
$$

（**关键**在于"取最小"这一操作对逐点 $(1\pm\varepsilon)$ 的扰动是稳定的，因此**最近邻集合在两空间中不同并不影响结论**——无需假设投影保持最近邻的身份。）

**(b)** 设 $\mathcal H_s\subseteq\mathcal H$ 为全空间 score 最小的 $s$ 个 honest 下标（$|\mathcal H_s|=s$，需 $s\le H$）。对 $i\in\mathcal H_s$，由 (14.17) 右半，

$$
\mathrm{sc}^{\mathrm{proj}}(i)\le(1+\varepsilon_{\mathrm{JL}})\,\mathrm{sc}^{\mathrm{full}}(i)
\le(1+\varepsilon_{\mathrm{JL}})\,\mathrm{sc}^{\mathrm{full},\mathcal H}_{(s)} .
$$

于是至少有 $s$ 个下标的投影 score 不超过 $(1+\varepsilon_{\mathrm{JL}})\mathrm{sc}^{\mathrm{full},\mathcal H}_{(s)}$，故全体投影 score 的第 $s$ 小者也不超过它。因 $\mathcal S_t$ 取投影 score 最小的 $s$ 个，对 $k\in\mathcal S_t$ 有 $\mathrm{sc}^{\mathrm{proj}}(k)\le(1+\varepsilon_{\mathrm{JL}})\mathrm{sc}^{\mathrm{full},\mathcal H}_{(s)}$。再由 (14.17) 左半，

$$
\mathrm{sc}^{\mathrm{full}}(k)\le\frac{\mathrm{sc}^{\mathrm{proj}}(k)}{1-\varepsilon_{\mathrm{JL}}}
\le\frac{1+\varepsilon_{\mathrm{JL}}}{1-\varepsilon_{\mathrm{JL}}}\,\mathrm{sc}^{\mathrm{full},\mathcal H}_{(s)}
=\theta_{\mathrm{JL}}\,\mathrm{sc}^{\mathrm{full},\mathcal H}_{(s)} . \qquad\square
$$

## 14.4 A5 成立：$\kappa_r$ 的显式表达式

> **定理 14.8（A5 的可实现性）**
> 在 Algorithm 1 中取 $\mathcal R_r=$ 投影空间 Multi-Krum（(14.16)，参数 $\hat B$、$s$），并设 A8（$N\ge2\hat B+3$）、$s\le H$、(14.5) 成立。则 **A5 成立**，且可取
> $$
> \boxed{\
> \kappa_r=\frac{2H}{\nu_H}\bigg(\frac{2\,\theta_{\mathrm{JL}}\,\nu H}{(H-1)(H-s+1)}+1\bigg),
> \qquad
> \nu=N-\hat B-2,\quad\nu_H=N-2\hat B-2 .\ }
> \tag{14.19}
> $$
> 特别地，取 $s=\lceil H/2\rceil$、$\hat B\le N/4$、$N\ge8$、$\varepsilon_{\mathrm{JL}}=\frac13$（$\theta_{\mathrm{JL}}=2$）时
> $$
> \kappa_r\le\frac{2H}{\nu_H}\Big(\frac{8\nu}{H-1}+1\Big)\le8\times(16+1)=136=O(1).
> \tag{14.20}
> $$

*证明.* 在事件 $\mathcal E_t$ 上，把命题 14.5 的证明中 Step 2 的 (14.15) 替换为引理 14.7(b) 的 (14.18)，即把 $\mathcal M$ 换成 $\theta_{\mathrm{JL}}\mathcal M$（其中 $\mathcal M$ 仍由 (14.14) 给出，它只涉及 honest 的**全空间** score，而 (14.14) 的推导（Step 1–2）完全在原空间进行，与投影无关）。Step 3、Step 4 逐字不变，得到：在 $\mathcal E_t$ 上确定性地

$$
\big\|\bar r_t^S-\bar r_t\big\|^2
\le\frac2{\nu_H}\big(\theta_{\mathrm{JL}}\mathcal M+H\,V_{r,t}\big)
=\kappa_r\,V_{r,t}
$$

（此处 $x_i\leftarrow r_t^{(i)}$，$\bar x_{\mathcal H}\leftarrow\bar r_t$，$V\leftarrow V_{r,t}$）。既然该不等式在 $\mathcal E_t$ 上逐样本成立，且 $V_{r,t}$ 为 $\mathcal F_t^{c}$-可测，取条件期望即得 A5 的 (3.4)：

$$
\mathbb E\Big[\big\|\bar r_t^S-\bar r_t\big\|^2\mathbf 1_{\mathcal E_t}\Big|\mathcal F_t^{c}\Big]
\le\kappa_rV_{r,t}\Pr\big(\mathcal E_t\mid\mathcal F_t^{c}\big)
\le\kappa_rV_{r,t} .
$$

(14.20)：在所给参数下 $\theta_{\mathrm{JL}}=2$，由命题 14.5 的数值验证，$\frac{2\theta_{\mathrm{JL}}\nu H}{(H-1)(H-s+1)}\le\frac{8\nu}{H-1}\le16$ 且 $\frac{2H}{\nu_H}\le8$。$\square$

> **注 14.9（这一步为什么成立：三个条件的缺一不可）**
> 1. **Condition 1（承诺）** 使 $\{r_t^{(i)}\}_{i\in[N]}$ 独立于 $V_t$，从而 (14.6) 对**含 Byzantine 的全部成对距离**成立。缺此则 §4 的攻击直接摧毁 (14.18)。
> 2. **Multi-Krum 只依赖成对距离**，使 (14.6) 足以推出 (14.17)；而"取最小"的稳定性使我们**不必**要求投影保持最近邻的身份。
> 3. **Multi-Krum 输出子集平均**，使投影空间的计算结果（下标集 $\mathcal S_t$）可以无损地"搬回"原空间：服务器广播 $\mathcal S_t$ 即可，(2.4) 的 $\bar P_t^S=\frac1{\sqrt r}\bar R_t^SV_t$ 由投影的线性性自动成立。

## 14.5 A7 成立：$\omega_{\mathrm{top}}$ 的显式表达式

> **定理 14.10（A7 的可实现性）**
> 在 Algorithm 1 中取 (2.5) 的 projected row score 与 Top-$k_{\mathrm{top}}$，并设 (14.5) 成立。则 **A7 成立**，事件族即 (14.4) 的 $\{\mathcal E_t\}$，$\delta_{\mathrm{JL}}$ 如推论 14.2，且
> $$
> \boxed{\
> \omega_{\mathrm{top}}
> =\frac{k_{\mathrm{top}}}{\theta_{\mathrm{JL}}\,(m-k_{\mathrm{top}})+k_{\mathrm{top}}}
> \ \ \ge\ \frac1{\theta_{\mathrm{JL}}}\cdot\frac{k_{\mathrm{top}}}m
> =\frac{1-\varepsilon_{\mathrm{JL}}}{1+\varepsilon_{\mathrm{JL}}}\cdot\frac{k_{\mathrm{top}}}m .\ }
> \tag{14.21}
> $$
> 以 $\gamma=k/m$、$k_{\mathrm{top}}=(1-\rho)k$ 表示即 $\omega_{\mathrm{top}}\ge\theta_{\mathrm{JL}}^{-1}(1-\rho)\gamma$，即 (13.3)。

*证明.* 在 $\mathcal E_t$ 上工作。记 $e_j:=\|(\bar R_t^S)_{j,:}\|^2$（$j\in[m]$），则 $\sum_je_j=\|\bar r_t^S\|^2=:\mathrm{Tot}$，且

$$
W:=\big\|(I-Q_t^{\mathrm{top}})\bar r_t^S\big\|^2=\sum_{j\notin I_t^{\mathrm{top}}}e_j .
$$

由 (2.5) 的定义，对任意 $j\in I_t^{\mathrm{top}}$ 与 $j'\notin I_t^{\mathrm{top}}$ 有 $\sigma_{t,j}\ge\sigma_{t,j'}$，配合 (14.7)：

$$
(1+\varepsilon_{\mathrm{JL}})e_j\ \ge\ \sigma_{t,j}\ \ge\ \sigma_{t,j'}\ \ge\ (1-\varepsilon_{\mathrm{JL}})e_{j'}
\quad\Longrightarrow\quad
e_j\ \ge\ \theta_{\mathrm{JL}}^{-1}e_{j'} .
$$

于是 $\min_{j\in I_t^{\mathrm{top}}}e_j\ge\theta_{\mathrm{JL}}^{-1}\max_{j'\notin I_t^{\mathrm{top}}}e_{j'}\ge\theta_{\mathrm{JL}}^{-1}\dfrac W{m-k_{\mathrm{top}}}$，故

$$
\mathrm{Tot}-W=\sum_{j\in I_t^{\mathrm{top}}}e_j
\ \ge\ k_{\mathrm{top}}\cdot\frac W{\theta_{\mathrm{JL}}(m-k_{\mathrm{top}})} .
$$

整理得 $\mathrm{Tot}\ge W\Big(1+\dfrac{k_{\mathrm{top}}}{\theta_{\mathrm{JL}}(m-k_{\mathrm{top}})}\Big)$，即

$$
W\ \le\ \frac{\theta_{\mathrm{JL}}(m-k_{\mathrm{top}})}{\theta_{\mathrm{JL}}(m-k_{\mathrm{top}})+k_{\mathrm{top}}}\,\mathrm{Tot}
=\big(1-\omega_{\mathrm{top}}\big)\mathrm{Tot}
$$

其中 $\omega_{\mathrm{top}}$ 如 (14.21)。这正是 A7 的 (3.7)。末端不等号由 $\theta_{\mathrm{JL}}(m-k_{\mathrm{top}})+k_{\mathrm{top}}\le\theta_{\mathrm{JL}}m$。$\square$

> **注 14.11** (14.21) 与原草稿的式 (27) 一致（原草稿给出的正是这里的下界），但本文给出了**精确值** $\frac{k_{\mathrm{top}}}{\theta_{\mathrm{JL}}(m-k_{\mathrm{top}})+k_{\mathrm{top}}}$，在 $k_{\mathrm{top}}$ 接近 $m$ 时明显更紧（如 $k_{\mathrm{top}}=m$ 时精确值为 $1$，而下界只给 $\theta_{\mathrm{JL}}^{-1}$）。
> 另外，(6.2) 保证最终 mask 满足 $\|(I-Q_t)\bar r_t^S\|^2\le\|(I-Q_t^{\mathrm{top}})\bar r_t^S\|^2$，故 **random refresh 不破坏 Top-K contraction**；而引理 6.1 的等式形式说明它进一步把因子乘上 $(1-p)$，这正是 (9.10) 的来源。

## 14.6 A6 成立：$\kappa_q$ 的显式表达式

tracker 侧的聚合在**全维**空间进行（服务器持有全部 $q_t^{(i)}\in\mathbb R^d$ 的同步副本），因此不涉及任何投影，可直接引用命题 14.5。

> **命题 14.12（A6 的可实现性）**
> 在 Algorithm 1 中取 $\mathcal A_q=\mathrm{MK}_{s_q}$（定义 14.4，参数 $\hat B$、输出规模 $s_q\le H$），并设 A8。则 **A6 成立**，且
> $$
> \boxed{\
> \kappa_q=\frac{2H}{\nu_H}\bigg(\frac{2\nu H}{(H-1)(H-s_q+1)}+1\bigg)\ }
> \tag{14.22}
> $$
> （无 $\theta_{\mathrm{JL}}$ 因子）。取 $s_q=\lceil H/2\rceil$、$\hat B\le N/4$、$N\ge8$ 时 $\kappa_q\le72=O(1)$。

*证明.* 直接对 $x_i\leftarrow q_t^{(i)}$、$\bar x_{\mathcal H}\leftarrow\bar q_t$、$V\leftarrow V_{q,t}$ 应用命题 14.5；(14.11) 是确定性不等式，故 A6 的 (3.5) 成立。$\square$

> **命题 14.12'（复合 ARC 之后）** 本版算法中 $\mathcal A_q=\mathrm{MK}_{s_q}\circ\mathrm{ARC}$。由推论 14.5a(b)、定理 10.4 与推论 10.7，
> $$
> \boxed{\
> \kappa_q^\star=\kappa\big(\cdot\big)+\frac{2\hat B}{N-2\hat B},
> \qquad
> \frac{2\hat B}{N-2\hat B}\le1\ \ (\hat B\le N/4)\ }
> \tag{14.22'}
> $$
> 其中 $\kappa(\cdot)$ 取 (14.11) 在相应参照子集下的值：参照子集大小为 $N-\hat B$ 时 $\le40$，参照子集为 $\mathcal H$ 时 $\le72$。故在 $\hat B\le N/4$、$N\ge8$ 下统一有
> $$
> \kappa_q^\star\ \le\ 73\ =\ O(1),
> $$
> 且在 $B=\hat B$（Byzantine 数量达到上界，最坏情形）时可收紧到 $\kappa_q^\star\le41$。

> **注 14.12a（缺口已闭合）** 变形A 初稿在此处标注过一个缺口：定理 10.4 建立在定义 10.5 的标准形式（对所有大小 $N-\hat B$ 的子集量化）之上，而原版命题 14.5 只对诚实集陈述。**本版已把命题 14.5 推广为「任意参照子集」形式并逐字给出证明**（见 §14.2），推论 14.5a 的两个用法分别兑现 A5 与定义 10.5，因此该缺口不复存在，且常数从 $579$ 改善到 $73$（注 14.5b）。原先用于弥补缺口的引理 10.6 降级为备用结果。

> **注 14.13（$\mathcal A_q$ 的计算复杂度与可替代性）**
> Multi-Krum 的 score 计算需要 $\binom N2$ 个 $d$ 维距离，即 $O(N^2d)$ 时间与 $O(N^2)$ 额外存储（距离矩阵）。在服务器侧、$N$ 为客户端数（通常 $N\ll d$）的联邦设定下这是可接受的，且与 tracker 副本的 $O(Nd)$ 存储同量级。
> 由于 A6 是纯粹的 $(f,\kappa)$-robustness，**$\mathcal A_q$ 可自由替换**为任何满足 (3.5) 的聚合器而不影响第 I 部分的任何结论——见 §14.7。

## 14.7 可替代的聚合器与 $\kappa$ 的改进

### 14.7.1 tracker 侧（A6）：几乎任何常见鲁棒聚合器均可

A6 是纯粹的 $(f,\kappa)$-robustness (3.5)，且 tracker 侧在全维空间操作，故任何满足该性质的聚合器都能直接代入，第 I 部分**一字不改**。文献中已知满足 (3.5) 的规则（$\kappa$ 取自 Allouah et al. 2023 的整理，$n=N$、$f=\hat B$）：

| $\mathcal A_q$ | $\kappa_q$ | 备注 |
|---|---|---|
| Multi-Krum（本文，命题 14.12） | (14.22)，$O(1)$ | 与 $\mathcal R_r$ 复用同一实现 |
| Krum（$s_q=1$） | $6\big(1+\frac{\hat B}{N-2\hat B}\big)$ | 无平均，方差缩减弱 |
| Coordinate-wise trimmed mean (CWTM) | $\frac{6\hat B}{N-2\hat B}\big(1+\frac{\hat B}{N-2\hat B}\big)$ | **随 $\hat B/N\to0$ 衰减** |
| Geometric median (RFA) | $4\big(1+\frac{\hat B}{N-2\hat B}\big)^2$ | 需迭代求解 |
| Coordinate-wise median | $4\big(1+\frac{\hat B}{N-2\hat B}\big)^2$ | $O(Nd)$，最快 |
| 任意上述规则 $\circ$ **NNM** 预聚合 | $O\big(\frac{\hat B}N\big)$ | **达到已知下界的阶** |

**推荐：$\mathcal A_q=\mathrm{CWTM}\circ\mathrm{NNM}$ 或 $\mathrm{MK}\circ\mathrm{NNM}$。** 由 §15.2，error floor 的主项 (I) 正比于 $\kappa_q$，而 NNM（nearest-neighbor mixing：先把每个 $q_t^{(i)}$ 替换为它的 $N-\hat B$ 个最近邻的平均，再送入聚合器）可把 $\kappa_q$ 从 $O(1)$ 降到 $O(\hat B/N)$，从而使 (I) 项

$$
O\Big(\frac{\kappa_q\Sigma^2}p\Big)
\ \longrightarrow\
O\Big(\frac{\hat B}N\cdot\frac{\Sigma^2}p\Big),
$$

与下界 $\Omega\big(\frac BN\zeta^2\big)$ 匹配到一个 $1/p$ 的压缩代价因子（§15.2）。NNM 的额外开销为 $O(N^2d)$，与 Multi-Krum 同量级。

### 14.7.2 correction 侧（A5）：两个结构性约束

$\mathcal R_r$ 的可选范围窄得多，因为它必须同时满足：

* **(C-a) 只依赖成对距离**（或其他 JL 可保持的量），否则无法在投影空间执行；
* **(C-b) 输出可由服务器仅广播"选择/权重信息"而在原空间无损重构**，否则第二阶段无法执行。

Multi-Krum 同时满足两者（注 14.9）。**不满足**的例子：几何中位数（输出不是输入的凸组合，且依赖全维几何）、CWTM 与 coordinate-wise median（逐坐标操作，投影后的坐标与原坐标无对应关系）。因此这两类只能用在 tracker 侧。

> **注 14.14（A5 可无代价地泛化到加权，从而容纳 NNM）**
> 检查第 I 部分对 $\bar r_t^S$ 的**全部**使用：仅 (i) A5 给出的 $\|\Delta_t\|^2$ 界（引理 9.3 的 (9.7)）与 (ii) A7 的 contraction (3.7) 作用在 $\bar r_t^S$ 上。二者都**不要求** $\bar r_t^S$ 是均匀子集平均。故可把 A5、(2.4) 泛化为
> $$
> \bar r_t^S:=\sum_{i=1}^Nw_{t,i}\,r_t^{(i)},
> \qquad
> w_t\ \text{为 }\mathcal G_t\text{-可测的概率向量}
> \quad\Big(w_{t,i}\ge0,\ \sum_iw_{t,i}=1\Big),
> \tag{14.23}
> $$
> 相应地 $\bar P_t^S=\frac1{\sqrt r}\bar R_t^SV_t$ 仍由线性性成立，(14.7) 取 $y=w_t$ 仍成立，$\mathcal E_t$ 与 $\delta_{\mathrm{JL}}$ 不变，第 I 部分的每一步逐字有效。
> 这使 **NNM 也可用于 correction 侧**：NNM 的最近邻选择只依赖成对距离（满足 (C-a)），其输出是输入的线性混合 $M_tx$（$M_t$ 行随机），服务器广播 $M_t$ 的行即可（满足 (C-b) 的加权版本）。此时 $\kappa_r$ 亦可降至 $O(\hat B/N)\cdot\theta_{\mathrm{JL}}$ 量级，使定理 B 的 (III) 项显著改善。为保持与原算法描述一致，正文主线仍取均匀子集平均。

## 14.8 三条假设的相容性

需要确认 A5、A6、A7 可以**同时**成立且参数选取无循环依赖。按以下顺序取值即可：

1. 取 $\varepsilon_{\mathrm{JL}}\in(0,1)$（如 $\frac13$），得 $\theta_{\mathrm{JL}}=\frac{1+\varepsilon_{\mathrm{JL}}}{1-\varepsilon_{\mathrm{JL}}}$（$=2$）。
2. 由 $k_{\mathrm{top}}$、$m$ 与 (14.21) 定出 $\omega_{\mathrm{top}}$（只依赖 $\theta_{\mathrm{JL}}$、$k_{\mathrm{top}}/m$，**不依赖 $r$**）。
3. 取 $\delta_{\mathrm{JL}}:=\omega_{\mathrm{top}}/4$，满足定理 12.2 的前提 $\delta_{\mathrm{JL}}\le\omega_{\mathrm{top}}/4$。
4. 由 (14.5) 定出 $r\ge c_0\varepsilon_{\mathrm{JL}}^{-2}\big(\min\{N,n\}+\log\frac{4m}{\omega_{\mathrm{top}}}\big)$。
5. 取 $s=s_q=\lceil H/2\rceil$（或按注 14.6 取更小的 $s$）；$k_{\mathrm{arc}}$ 由 (2.0) 唯一确定，无需选取。

**无循环依赖**：$\omega_{\mathrm{top}}$ 只依赖 $\varepsilon_{\mathrm{JL}}$ 与行预算，$\delta_{\mathrm{JL}}$ 只依赖 $\omega_{\mathrm{top}}$，$r$ 只依赖前两者。三条假设由定理 14.8、命题 14.12、定理 14.10 分别兑现，且 A5 与 A7 使用的是**同一个** $\mathcal E_t$（(14.4)），满足 A7 对"共用事件族"的要求。$H$ 未知不影响可执行性：$s,s_q$ 可用 $\hat H:=N-\hat B$ 代替 $H$，此时 $s\le\hat H\le H$ 仍满足命题 14.5 的前提；$k_{\mathrm{arc}}$ 只依赖 $N$ 与 $\hat B$，本就与 $H$ 无关。

## 14.9 端到端实例化结论

> **推论 14.15（实例化后的收敛保证）**
> 在 Algorithm 1' 中取 $\mathcal R_r=$ 投影空间 Multi-Krum、$\mathcal A_q=\mathrm{MK}_{s_q}\circ\mathrm{ARC}$，参数按 §14.8 取值，$\hat B\le N/4$、$N\ge8$、$s=s_q=\lceil H/2\rceil$、$\varepsilon_{\mathrm{JL}}=\frac13$，步长按推论 12.7 取 $\eta=\min\{\eta_{\max},\eta_0T^{-1/2}\}$。则 A1–A9 全部成立，$\kappa_q^\star\le73$（$B=\hat B$ 时 $\le41$）、$\kappa_r\le136$、$\omega_{\mathrm{top}}\ge\frac13\cdot\frac{k_{\mathrm{top}}}m$、$\delta_{\mathrm{JL}}=\frac{\omega_{\mathrm{top}}}4$，且
> $$
> \frac1T\sum_{t=0}^{T-1}\mathbb E\|\nabla f(\theta_t)\|^2
> =O\Big(\frac1{\sqrt T}\Big)
> +\mathcal F_{\mathrm{floor}} ,
> \tag{14.24}
> $$
> $$
> \mathcal F_{\mathrm{floor}}
> =O\bigg(\underbrace{\frac{\kappa_q^\star\zeta^2}p}_{\text{(I)}}
> +\min\bigg\{
> \underbrace{0}_{\text{定理 A}},\ \
> \underbrace{\frac{\kappa_r\zeta^2}{\omega_{\mathrm{top}}^2}\Big(1+\frac1{\sqrt p}\Big)^2}_{\text{定理 B}}
> \bigg\}\bigg),
> \tag{14.25}
> $$
> 梯度复杂度 $T=O(\varepsilon^{-4})$（推论 12.5）。若进一步取 $\mathcal A_q=\mathrm{CWTM}\circ\mathrm{NNM}\circ\mathrm{ARC}$，则 (I) 项改善为 $O\big(\frac{\hat B}N\cdot\frac{\zeta^2}p\big)$，**与下界 $\Omega(\frac BN\zeta^2)$ 只差一个压缩代价因子 $1/p$**。
>
> **与变形A 的差别**：$\mathcal F_{\mathrm{floor}}$ 中**不再含 $\sigma^2$**——全部采样噪声项经 $1-\beta=\Theta(T^{-1/2})$ 退化为 $O(T^{-1/2})$（推论 12.4）。剩下的只有异质性 $\zeta^2$，而它有匹配下界。
> **与完整版的差别**：右端还**不再有 $O(\mathcal B_{\mathrm{clip}}^{(T)})$**，(14.24) 完全闭合。

### 14.9.1 两个主定理的适用区间（重要）

(14.25) 的 $\min$ 不是形式化的：定理 A 与定理 B 各有其占优区间。以 $p\approx\rho\gamma$、$\omega_{\mathrm{top}}\approx\theta_{\mathrm{JL}}^{-1}(1-\rho)\gamma$ 代入：

* **$\rho=\Theta(1)$（如 $\rho^\star\approx\frac13$）时，$p$ 与 $\omega_{\mathrm{top}}$ 同阶，$\chi\approx p$。** 此时定理 B 的 contraction 相对定理 A 无实质改进，却多出 $O(\kappa_r\Sigma^2/(p\,\omega_{\mathrm{top}}^2))=O(\Sigma^2/\gamma^3)$ 项，**定理 A 更优**。
* **$p\ll\omega_{\mathrm{top}}$（$\rho$ 很小、行预算主要给 Top-K）时，$\chi\approx\omega_{\mathrm{top}}/4\gg p$。** 定理 A 的 $\frac{\sigma^2}{Hp^2}$ 急剧变差，而定理 B 用 $\frac{\sigma^2}{H\omega_{\mathrm{top}}^2}$ 替代。定理 B 占优的充分条件是
$$
\frac{\kappa_r\Sigma^2}{\omega_{\mathrm{top}}^2\,p}\ \lesssim\ \frac{\sigma^2}{Hp^2}
\quad\Longleftrightarrow\quad
p\ \lesssim\ \frac{\omega_{\mathrm{top}}^2\,\sigma^2}{H\,\kappa_r\,\Sigma^2},
\tag{14.26}
$$
 即 refresh 比例足够小、且 correction 侧选择足够准确（$\kappa_r$ 小）时。

**结论：两个定理是互补的，不是强弱关系。** 推论 13.1 的 $\rho^\star$ 是"以定理 B 的界为目标"的最优比例；若以 $\min$ 为目标，则应在 $\rho=\Theta(1)$（走定理 A）与 $\rho$ 满足 (14.26)（走定理 B）两个候选中取更优者。这也回答了原草稿"问题 §1"更深一层的疑问：把 $\kappa_r,\omega_{\mathrm{top}}$ 写进定理是可以做到的（定理 12.2），但**它们的引入并非无代价**——correction 侧的鲁棒性误差 $\kappa_r$ 会以 $\kappa_r/\omega_{\mathrm{top}}^2$ 的形式进入 error floor，只有在该项被压住时精细定理才真正更强。

---

# 15. 与文献的定位

## 15.1 复杂度对比

| 算法 | 梯度类型 | 压缩 | 复杂度 | error floor | 需 Hessian 方差假设 | 需有界梯度 |
|---|---|---|---|---|---|---|
| Byz-VR-MARINA (NeurIPS'23) | 有限和 + 周期全梯度 | 无偏 | $O(\varepsilon^{-2})$ | $O(\kappa\tau^2)$ | **是** | 否 |
| Byz-DASHA-PAGE (NeurIPS'24) | 有限和 + 周期全梯度 | 无偏 | $O(\varepsilon^{-2})$ | $O(\frac{\kappa\tau^2}{1-c\kappa})$ | **是** | 否 |
| Byz-EF21 (NeurIPS'24) | **全梯度** | 有偏 Top-$k$ | $O(\varepsilon^{-2})$ | $O(\kappa(1+\sqrt\kappa)\zeta^2)$ | **是** | 否 |
| RoSDHB (AISTATS'26) | 随机 + HB 动量 | 无偏 RandK | $O(\varepsilon^{-2})$ | $O(\frac{\kappa G^2}{1-\kappa B^2})$ | 否 | **是**（$G^2$） |
| Byz-VR-DM21 (AISTATS'26) | 随机 + 本地 VR | 有偏 + 双动量 | $O(\varepsilon^{-3})$ | $\kappa\zeta^2$ | 否 | 否 |
| Byz-EF21-SGDM (TNNLS'26) | 随机 | 有偏 + EF21 | $O(\varepsilon^{-4})$ | $O(\sigma^2)+18\kappa G^2$ | 否 | **是**（$G$） |
| Byz-DM21 (AISTATS'26) | 随机 | 有偏 + 双动量 | $O(\varepsilon^{-4})$ | $O(\sigma^2)+32\kappa\zeta^2$ | 否 | 否 |
| **本文 Algorithm 1''（变形A + 客户端动量）** | **随机（无全梯度）** | **投影 + 有偏 Top-$k$ 行** | $O(\varepsilon^{-4})$ | (14.25) $=O(\frac{\kappa_q^\star\zeta^2}p)$，**纯异质性、无 $\sigma^2$、无未闭合项** | **否** | **否**（全文，无可选条件） |
| Adam 侧参照：Reddi et al. (ICLR'21) | 随机 | 无 | — | — | 否 | **是**（逐坐标、几乎必然、对随机梯度） |
| Adam 侧参照：Li–Rakhlin–Jadbabaie (2023) | 随机 | 无 | $O(\varepsilon^{-4})$ | — | 否 | 否，但需**几乎必然有界/次高斯噪声**且结论为高概率型 |

**定位。** 本算法属于第四梯队（纯随机梯度、无方差缩减），复杂度 $O(\varepsilon^{-4})$ 与 Byz-EF21-SGDM、Byz-DM21 同阶，这是该设定的标准阶。其区别性贡献不在复杂度阶，而在：

1. **两阶段投影压缩 + 全局一致 mask**：第一阶段只上传 $mr$ 个数即可让服务器在**低维**完成鲁棒子集选择与行打分，第二阶段上传 $kn$ 个数。与逐客户端独立 Top-$k$（各客户端 mask 不同、honest averaging 与 tracker 更新不可交换）相比，全局一致 mask 使 (7.3)、(7.4) 的正交分解成立，这是 dispersion 闭合的前提。
2. **Adam + Byzantine + 压缩的完整分析**：FedAdam-BACK 在其 Table 1 中把自身速率标为"??"；本文给出 Adam 型自适应更新在拜占庭鲁棒压缩下的显式**且闭合**的速率，preconditioner 界不依赖任何有界性假设（注 3.2、引理 11.1）。
3. **不需要全局 Hessian 方差假设**（区别于 Byz-VR-MARINA / Byz-DASHA-PAGE / Byz-EF21），**也不需要有界梯度**（区别于 RoSDHB、Byz-EF21-SGDM），**且不需要几乎必然有界或次高斯噪声**（区别于 Li–Rakhlin–Jadbabaie 2023、Hong–Lin 2023 这条去掉有界梯度的 Adam 主线）。就假设强度而言，本文与 Byz-DM21 相当，而在 Adam 这一维上严格弱于全部已知的自适应联邦优化分析。
   这一点的实现方式是**结构性的而非分析技巧**：把 cap 限制在二阶矩累加器上，使 Adam 所需的 preconditioner 谱界成为算法的构造性事实，同时更新方向上不含任何裁剪算子。注 11.8a 说明，若不做这一改动，在只有期望型有界方差时 clipping bias **必然**无法闭合——因此这不是可选的优化。
4. **random refresh 机制及其 universal contraction**：注 6.2、注 7.7 说明它解决了"有偏 Top-K + stateful tracker + Byzantine"三者叠加下 dispersion 可能发散的问题，这是文献中未见的处理。
5. **三个机制的相容性**：客户端动量（压小 dispersion 的噪声成分）、ARC 预聚合裁剪（把裁剪吸收进鲁棒常数）、cap 只作用于二阶矩（构造性给出 Adam 谱界）——三者各自改动算法的不同环节，且在同一个自洽闭合（定理 11.15）中一并处理。三者叠加之后，主定理**同时**做到：闭合、只用有界方差、error floor 纯异质性。据我们所知这个组合在文献中没有对应物。

## 15.2 error floor 与下界的距离

已知下界（Karimireddy–He–Jaggi 2022；Allouah et al. 2023）：在 $(B,\zeta^2)$-异质性下，任何算法的稳态误差满足

$$
\liminf_{T\to\infty}\ \frac1T\sum_{t}\mathbb E\|\nabla f(\theta_t)\|^2
\ \ge\ \Omega\Big(\frac BN\,\zeta^2\Big).
\tag{15.1}
$$

本文 (14.25) 的主项为 $O(\kappa_q\Sigma^2/p)$。逐因子对照：

| 因子 | 来源 | 是否可消除 |
|---|---|---|
| $\zeta^2$（含于 $\Sigma^2$） | 数据异质性（A4） | **否**，(15.1) |
| $\kappa_q$ | 聚合器质量 | 可从 $O(1)$（Multi-Krum）降至 $O(\hat B/N)$（+NNM），**达到 (15.1) 的阶** |
| $\sigma^2$ | 随机梯度噪声 | **已消除**：客户端动量使其全部带因子 $(1-\beta)$，取 $1-\beta=\Theta(T^{-1/2})$ 后以 $O(T^{-1/2})$ 消失（推论 12.4）。这是本版相对变形A 的核心改进 |
| $1/p$ | **压缩代价**：$\bar V_q^{(T)}\le\Sigma_\beta^2/p$（定理 7.4） | 本文未消除；$p\to1$（无压缩）时退化为无此因子 |

因此结论是：**在客户端动量消去全部 $\sigma^2$ 项、且 $\mathcal A_q$ 采用 NNM 加强之后，本文的 error floor 为 $O\big(\frac{\hat B}N\cdot\frac{\zeta^2}p\big)$，与下界 (15.1) 相差恰好一个压缩代价因子 $1/p=\Theta(1/\gamma)$。**
这是本工作理论部分的最终位置：**error floor 与已知下界的全部差距只剩压缩代价这一个因子**。它是否可以改进（例如降到 $\log\frac1p$ 或常数）是留下的主要理论问题，见 §16 第 1 条。

---

# 16. 局限与开放问题

1. **$1/p$ 压缩代价因子（本工作留下的首要理论问题）。** 定理 7.4 的 $\bar V_q^{(T)}\le\Sigma_\beta^2/p$ 是 error floor 的公共来源，在 $p\to0$（强压缩）时发散。**在客户端动量消去全部 $\sigma^2$ 项之后，这已是 error floor 与已知下界 $\Omega(\frac BN\zeta^2)$ 之间的唯一差距**（注 12.6、§15.2）。直观上刷新概率 $p$ 意味着每个 tracker 行平均每 $1/p$ 轮更新一次，其 staleness 引起的 dispersion 放大 $1/p$ 倍似乎是本质的；但相应下界尚未建立。这是最值得做的后续工作。

2. **$O(\varepsilon^{-4})$ 未突破。** 本算法为纯随机梯度、无方差缩减，故复杂度为该设定的标准阶。引入本地方差缩减（如 Byz-VR-DM21 的做法）有望降到 $O(\varepsilon^{-3})$。技术障碍在于：方差缩减的估计量是历史梯度的差分，与 EF21 tracker 的 staleness 及客户端动量三者叠加后，(7.7) 的正交分解不再直接可用。
   另一条更容易的路线：若客户端每轮使用**全部本地数据**（$\sigma=0$，cross-silo 场景），则 (11.11) 的驱动项只剩 $\eta^2$ 项，$\bar E$ 随步长趋零，可取常数步长得 $O(1/T)$ 的优化项，即 $T=O(\varepsilon^{-2})$。此时与 Byz-EF21 (NeurIPS'24) 同阶，但**不需要其全局 Hessian 方差假设**。该推论本文未展开，但只需把 $\sigma=0$ 代入 §11–§12 即可，值得作为一个独立结果补上。

3. **~~clipping bias~~ 与 ~~噪声 floor~~ 均已解决。** 完整版遗留的 $\mathcal B_{\mathrm{clip}}^{(T)}$ 在变形A 中已不存在（注 11.8 的机制）；变形A 遗留的 $\sigma^2/(Hc^2)$ 在本版中经客户端动量退化为 $O(T^{-1/2})$（推论 12.4）。取而代之的是本版自身的三个开放问题：

   **3a. 步长上限的两个收紧因子。** (11.18) 要求 $\eta\lesssim c\cdot\frac{\lambda_A}{L\Lambda_A^2}$（来自变形A 解出 $\bar Z$ 的自洽步骤）与 $\eta\lesssim(1-\beta)$（来自客户端动量偏差，注 11.16）。前者使 burn-in 门槛为 $\Theta(c^{-2})$，后者在取 $1-\beta=\Theta(T^{-1/2})$ 时与 $\eta$ 同阶因而不额外收紧，但二者都体现为界中较大的常数。是否可以通过更细的自洽论证（例如对 $\bar Z$ 用带权重的递推而非一次性放大）去掉这些因子，尚未清楚。

   **3b. 界看不到自适应性。** 注 13.1 指出 $C_A^\sharp$ 关于 $C_{\max}$ 单调递增且无抵消项，故本文的界对 SGDM（$C_{\max}\to0$）给出的常数总是不劣于对 Adam 给出的常数。这是全部最坏情况 Adam 分析的共同局限。要让理论真正体现自适应性的好处，需要引入坐标级的结构假设（如梯度尺度的坐标间异质性），这超出本文范围。

   **3c. $\Sigma_\beta^2$ 中 $\zeta^2$ 的常数 $2$。** (7.4) 用 Minkowski 后再用 $(a+b)^2\le2a^2+2b^2$，使异质性项带常数 $2$。用带参数的 Young（取参数随 $1-\beta$ 变化）可以把它改善到 $1+O(\sqrt{1-\beta})$，即 $\beta\to1$ 时趋于 $1$。本文未做这一优化以保持表达简洁，但正式版应补上——因为 $\zeta^2$ 的系数正是与下界对照时唯一要看的量。

4. **$\kappa_r$ 的代价。** §14.9.1 表明定理 B 并非在所有参数区间优于定理 A：$\kappa_r$ 以 $\kappa_r/\omega_{\mathrm{top}}^2$ 进入 floor。改进方向是把引理 9.3 中 $\Delta_t$ 的系数 $10/\omega_{\mathrm{top}}$ 降低——它来自 (Y) 的两次使用，若能利用 $\Delta_t$ 与 $\bar r_t$ 的（部分）正交性或对 $\Delta_t$ 的方向做更细的分解，可能改进到 $O(1)$。

5. **投影维数中的 $\min\{N,n\}$。** (14.5) 的 $\min\{N,n\}$ 来自 subspace embedding 对整个 $N$ 维系数空间的一致保证。若只需覆盖 Multi-Krum 实际用到的 $O(N^2)$ 个成对距离与 $O(1)$ 个子集平均，标准 JL 只需 $O(\varepsilon^{-2}\log\frac{mN}\delta)$；难点在于 $\mathcal S_t$ 依赖 $V_t$，使"实际用到的 $y$"不能事先确定。一个可能的改进是对 selector 做 stability 分析（证明 $\mathcal S_t$ 只在少数候选集之间变动），从而把 union 范围从 $\binom Ns$ 降到多项式规模。

6. **部分参与（partial participation）。** 本文假设每轮全体客户端参与。部分参与下 tracker 的 staleness 与 refresh 概率耦合，需要重新推导 (7.5)；D-Byz-SGDM 的 delayed momentum aggregation 提供了实验层面的思路，但尚无理论。

7. **$\varepsilon_{\mathrm{JL}}$ 与 $\omega_{\mathrm{top}}$ 的联合优化。** (14.21) 表明 $\omega_{\mathrm{top}}$ 随 $\varepsilon_{\mathrm{JL}}$ 减小而增大，而 (14.5) 表明 $r\propto\varepsilon_{\mathrm{JL}}^{-2}$。§13.4 取 $\varepsilon_{\mathrm{JL}}=\frac13$ 是一个方便的折中，最优取值应由 (14.24) 与通信预算 (2.11) 的联合优化决定。

---

# 参考文献

1. Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. *NeurIPS 2017*.（Krum / Multi-Krum）
2. Karimireddy, S. P., He, L., & Jaggi, M. (2021). Learning from History for Byzantine Robust Optimization. *ICML 2021*.（agnostic robust aggregator）
3. Karimireddy, S. P., He, L., & Jaggi, M. (2022). Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing. *ICLR 2022*.（异质性下界与 bucketing）
4. Allouah, Y., Farhadkhani, S., Guerraoui, R., Gupta, N., Pinot, R., & Stephan, J. (2023). Fixing by Mixing: A Recipe for Optimal Byzantine ML under Heterogeneity. *AISTATS 2023*.（$(f,\kappa)$-robustness、NNM、各聚合器的 $\kappa$）
5. Richtárik, P., Sokolov, I., & Fatkhullin, I. (2021). EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback. *NeurIPS 2021*.
6. Sarlós, T. (2006). Improved Approximation Algorithms for Large Matrices via Random Projections. *FOCS 2006*.（subspace embedding）
7. Woodruff, D. P. (2014). Sketching as a Tool for Numerical Linear Algebra. *Foundations and Trends in TCS*, 10(1–2).
8. Vershynin, R. (2018). *High-Dimensional Probability*. Cambridge University Press.（Thm 4.6.1）
9. Zhu, F., & Ling, Q. (2023). BROADCAST: Reducing Both Stochastic and Compression Noise to Robustify Communication-Efficient Federated Learning. *IEEE TSIPN*, 9, 280–294.
10. Gorbunov, E., et al. (2023). Byzantine-Robust Variance-Reduced Federated Learning. *NeurIPS 2023*.
11. Rammal, A., et al. (2024). Byzantine-Resilient Decentralized SGD with Compression. *NeurIPS 2024*.（Byz-DASHA-PAGE、Byz-EF21）
12. Liu, Y., Li, X., Yi, J., & Johansson, M. (2026). Byzantine-Robust Federated Learning with Biased Gradient Compression. *IEEE TNNLS*.
13. Li, X., Liu, Y., & Yi, J. (2026). Double Momentum for Byzantine-Robust Decentralized Learning with Biased Compressors. *AISTATS 2026*.（Byz-DM21、Byz-VR-DM21）
14. Gupta, A., Honsell, F., Xu, X., Gupta, H., & Neglia, G. (2026). Robust Stochastic Decentralized Learning with Heavy Ball. *AISTATS 2026*.（RoSDHB）
15. Défossez, A., Bottou, L., Bach, F., & Usunier, N. (2022). A Simple Convergence Proof of Adam and Adagrad. *TMLR*.（Adam 分析，假设随机梯度有界）
16. Allouah, Y., Guerraoui, R., Gupta, N., Jellouli, A., Rizk, G., & Stephan, J. (2025). Adaptive Gradient Clipping for Robust Federated Learning. *ICLR 2025*.（**ARC**：定义 10.1、定理 10.4、引理 10.8；以及静态裁剪破坏鲁棒性的 Lemma 3.1）
17. Koloskova, A., Hendrikx, H., & Stich, S. U. (2023). Revisiting Gradient Clipping: Stochastic Bias and Tight Convergence Guarantees. *ICML 2023*.（裁剪偏差的紧上下界 $\min\{\sigma,\sigma^2/c\}$，一次幂形式；注 11.8a 的背景）
18. Li, H., Rakhlin, A., & Jadbabaie, A. (2023). Convergence of Adam Under Relaxed Assumptions. *NeurIPS 2023*.（去掉有界梯度，代价为几乎必然有界/次高斯噪声 + 高概率结论）
19. Hong, Y., & Lin, J. (2024). High Probability Convergence of Adam Under Unbounded Gradients and Affine Variance Noise.（同上路线，逐坐标 affine variance 且几乎必然成立）
20. Reddi, S. J., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., & McMahan, H. B. (2021). Adaptive Federated Optimization. *ICLR 2021*.（FedAdam/FedYogi/FedAdagrad；假设逐坐标、几乎必然有界的随机梯度）
21. Gratton, S., & Toint, Ph. L. (2026). A unified convergence theory for adaptive first-order methods in the nonconvex case.（自适应方法统一框架；其梯度预言机假设"necessarily stronger than bounded variance"）

---

*本文档的第 I 部分（§1–§13）不依赖任何具体聚合规则；第 II 部分（§14）证明其抽象假设可由"投影空间 Multi-Krum + ARC + 单套 Gaussian 投影 + projected Top-K"同时实现。两部分可独立阅读与独立替换。*

---

## 附：本版（变形A + 客户端动量）相对变形A 的改动清单

**算法（一处）**
* §2.2 第 1 步：新增客户端动量缓存 (2.1a)，$u_0^{(i)}=g_0^{(i)}$、$u_t^{(i)}=\beta u_{t-1}^{(i)}+(1-\beta)g_t^{(i)}$。
* §2.2 第 2 步与第 9 步：correction 与 tracker 改为针对 $u_t^{(i)}$。
* §2.1：新增参数 $\beta$。
* **通信量、通信轮次、压缩率、服务器计算全部不变**；客户端每轮多存一个 $d$ 维缓存、多一次向量线性组合。$\beta=0$ 时精确退化为变形A。

**假设：完全不变**
* A1–A9 与变形A 逐字相同，仍施加在原始随机梯度 $g_t^{(i)}$ 上（注 3.0）。动量缓存的性质是**推导**出来的，不是假设。

**推导：完全未改动的部分**
* §4（承诺机制）、§5（filtration）、§6（两条 contraction）、§10（ARC）、§14（可实现性）。
* §9 的全部论证：把被追踪对象由 $\bar g_t$ 换成 $\bar u_t$ 之后**逐字成立**——引理 9.1 的代数、引理 9.3 的 good-event 处理、定理 9.5 的乘性叠加、引理 9.7 的时间平均，均不依赖被追踪对象的具体来源。
* §11.1（preconditioner）、§11.2（一阶矩的二阶矩）、§11.6（服务器动量残差）、§11.7（单步下降）与变形A 逐字相同。

**推导：重写的部分**
* **§7 全节**：新增引理 7.1（动量 dispersion：噪声乘 $\frac{1-\beta}{1+\beta}$、异质性不变），定理 7.4/7.6 改为**时间平均**形式并把 $\Sigma^2$ 换成 $\Sigma_\beta^2$。改用时间平均是为了处理 $\varrho_t$ 的预热暂态，常数无损失（注 7.5）。
* **§11.3 新增**：引理 11.4 与推论 11.5，闭合客户端动量偏差 $\mathcal B_t=\mathbb E\|\bar u_t-h_t\|^2$。这是引入动量的唯一新代价。
* **§11.5 重写**：引理 11.9，漂移的 $\sigma^2/H$ 系数由 $4$ 变为 $6(1-\beta)^2$。这是引入动量的全部收益来源。
* **§11.4、§11.8**：estimator 误差的末项由 $3\sigma^2/H$ 换成 $3\mathcal B_t$；自洽闭合中 $\theta_c$ 新增 $c_\beta^{\mathrm{cl}}$ 项，步长条件新增 $\eta\lesssim1-\beta$。
* **§12**：主定理的 floor 拆成"与 $\beta$ 无关的 $\zeta^2$ 部分"与"正比于 $(1-\beta)$ 的 $\sigma^2$ 部分"（推论 12.3），并新增联合取参推论 12.4。

**结论的变化**
* 变形A：$\mathcal F_{\mathrm{floor}}=O\big(\frac{\kappa_q^\star\Sigma^2}p+\frac{\sigma^2}{Hc^2}\big)$，其中 $\sigma^2$ 项不随任何参数消失。
* 本版：$\mathcal F_{\mathrm{floor}}=O\big(\frac{\kappa_q^\star\zeta^2}p\big)$（定理 A）或再加 $O\big(\frac{\kappa_r\zeta^2}{\omega_{\mathrm{top}}^2}(1+p^{-1/2})^2\big)$（定理 B），**纯异质性**。
* 与下界 $\Omega(\frac BN\zeta^2)$ 的差距：聚合器常数（可用 NNM 匹配）+ 压缩代价 $1/p$（开放问题 §16.1）。

**三个版本的关系**

| | 完整版 | 变形A | 本版 |
|---|---|---|---|
| 裁剪 | 后聚合 capped adaptive | 预聚合 ARC | 同变形A |
| cap 作用位置 | 更新方向 + 二阶矩 | **仅二阶矩** | 同变形A |
| clipping bias | **未闭合**（留 $\mathcal B_{\mathrm{clip}}^{(T)}$） | 不存在 | 不存在 |
| 可选条件 A10'（有界梯度） | 需要（为使 bias 为零） | 已删除 | 已删除 |
| tracker 追踪对象 | $g_t^{(i)}$ | $g_t^{(i)}$ | **$u_t^{(i)}$（动量缓存）** |
| floor 的 $\sigma^2$ 项 | $O(\frac{\sigma^2}{Hp^2})$，不消失 | 同 | **$O((1-\beta)\sigma^2)$，可消失** |
| floor 的最终形态 | $O(\frac{\Sigma^2}p+\frac{\sigma^2}{Hp^2})+\mathcal B_{\mathrm{clip}}$ | $O(\frac{\Sigma^2}p+\frac{\sigma^2}{Hp^2})$ | $O(\frac{\kappa_q^\star\zeta^2}p)$ |
| 步长条件 | $\eta\lesssim\frac{\lambda_A}{L\Lambda_A^2}$ | $+\ \eta\lesssim c$ | $+\ \eta\lesssim1-\beta$ |
