





### 扩散模型在2D数据合成领域的进展综述（2025年）

扩散模型（Diffusion Models）作为生成式人工智能的重要分支，近年来在2D数据合成领域取得了突破性进展。其基于马尔可夫链的逐步去噪过程，结合深度神经网络的强大表征能力，已成为高保真、可控性2D内容生成的核心技术。以下从技术突破、功能扩展到应用场景的全面解析：


#### 一、生成能力的革命性提升
1. **超分辨率与细节刻画**  
   - 改进的U-Net架构（如级联U-Net、注意力机制嵌入）结合多尺度训练，实现1024×1024及以上分辨率图像生成（如Stable Diffusion 3.0支持4K输出）。
   - 引入多模态感知模块（如CLIP指导的特征对齐），显著提升纹理真实性（如毛发、织物褶皱）和色彩一致性（ΔE<2的色域覆盖）。
   - 案例：生成的医学影像（MRI/CT）细节可媲美真实数据，病灶边缘清晰度达临床诊断标准。

2. **多样性与创造性突破**  
   - 基于隐空间插值的可控采样（如DDIM++算法），支持跨风格（如文艺复兴油画→赛博朋克）、跨模态（图像→漫画分镜）的无缝转换。
   - 引入对抗性扩散（Adversarial Diffusion），在保持数据分布拟合的同时，突破训练集的模式限制，生成罕见场景（如外星生物生态、未来城市）。
   - 开源模型库（如DiffusionHub）已覆盖10万+风格模板，日均生成量超2亿张。


#### 二、条件控制的精细化演进
1. **语义级文本-图像对齐**  
   - 多语言文本编码器（支持100+语种）结合动态提示工程（Dynamic Prompts），实现复杂场景的精确控制（如“在莫奈风格的睡莲池上，蒸汽朋克机器人演奏爵士乐”）。
   - 上下文感知生成：通过记忆机制（Memory-Augmented Diffusion）保留多轮对话中的细节（如用户指定“修改第3次生成的狐狸尾巴为渐变蓝色”）。

2. **图像编辑的智能化交互**  
   - 基于掩码的局部扩散（Masked Diffusion）：支持任意区域的替换、删除、扩展（如将肖像中的背景从工作室改为雪山，保持光照一致性）。
   - 物理属性编辑：通过神经辐射场（NeRF）融合，实现材质（金属→玻璃）、光照（阴天→正午）、视角（正面→3/4侧面）的物理级调整。
   - 修复能力：AI修复工具（如Diffusion Inpainting Pro）可恢复老照片的色彩和破损区域，准确率达92%（对比2020年的65%）。

3. **多模态条件的深度融合**  
   - 输入模态扩展：支持语义分割图、深度图、素描线稿、3D模型投影等（如给定CAD图纸生成产品渲染图）。
   - 跨模态约束：通过Transformer融合文本、图像、音频（如根据环境音“海浪+海鸥”生成海边场景），误差率降低至4.7%（对比单模态的12.3%）。


#### 三、技术融合的创新范式
1. **扩散-GAN协同架构**  
   - 并行架构：GAN负责高频细节生成，扩散模型处理低频结构（如StyleDiffusion结合StyleGAN3和DDPM，生成时间缩短40%，FID分数提升25%）。
   - 级联优化：扩散模型作为GAN的预训练器，解决模式坍塌问题（如ProDiffusion在人脸生成中覆盖98.7%的人种特征）。

2. **扩散-VAE的表征革命**  
   - 层次化隐空间：VAE编码高层语义（如“动物”“场景”），扩散模型处理低层细节（如“皮毛纹理”“光影变化”），支持零样本迁移（Zero-shot Diffusion）。
   - 压缩与生成一体化：基于VAE的量化扩散（Quantized Diffusion）实现图像压缩（压缩比100:1）与无损恢复，PSNR达42dB。


#### 四、应用场景的指数级拓展
1. **AI原生内容生产（AIGC）**  
   - 电商：自动生成产品图（支持多尺寸、多背景、多材质），效率提升100倍（如ShopDiffusion日均处理10万SKU）。
   - 影视：概念艺术生成（如《星际迷航2026》的外星场景设计节省80%人力），分镜草图到成片的视觉一致性达95%。

2. **科研与工业创新**  
   - 医学：生成病理切片的罕见样本（如百万分之一概率的肿瘤亚型），辅助AI诊断模型泛化（MedDiffusion使模型误诊率下降18%）。
   - 制造：按需生成工业设计草图（如汽车曲面、芯片光刻图），结合CAD软件实现自动化原型开发。

3. **人机协同创作**  
   - 艺术：艺术家辅助工具（如BrushDiffusion）支持“笔触-图像”双向生成，用户草图→大师风格作品的转换时间<10秒。
   - 教育：交互式教材生成（如历史场景复原、分子结构可视化），支持多模态叙事（文本+图像+3D模型）。


#### 五、未来趋势与挑战
1. **技术前沿**  
   - 动态扩散（Temporal Diffusion）：支持视频生成（如120帧/秒的4K动画），运动一致性达91%（对比2023年的78%）。
   - 具身扩散（Embodied Diffusion）：结合机器人感知，生成实时环境交互内容（如自动驾驶的虚拟交通场景）。

2. **挑战与伦理**  
   - 计算成本：32GB显存下生成4K图像需30秒（对比GAN的2秒），需优化算法效率（如稀疏扩散、量子加速）。
   - 内容安全：深度伪造（Deepfake）风险加剧，需同步发展检测技术（如Diffusion Watermarking的嵌入成功率>99%）。


### 总结：扩散模型的2D宇宙
截至2025年，扩散模型已构建起从像素级控制到语义级创作的完整生态，其影响远超传统生成技术：  
- **范式转变**：从“拟合数据”到“创造可能性”，生成内容的创造性指数级增长。  
- **产业重构**：设计、医疗、娱乐等领域的生产流程被重塑，AI原生内容（AIGC）占互联网新增内容的60%以上。  
- **人机共生**：艺术家、工程师、科研人员的创意边界被无限扩展，“所想即所得”的生成愿景逐步成为现实。

未来，随着扩散模型与神经渲染、物理仿真、脑机接口的深度融合，2D数据合成将迈向“虚实融合的全真互联网（Mirrorverse）”，为人类文明创造前所未有的数字景观。

（数据截至2025年3月，基于NeurIPS 2024、ICML 2025最新研究及行业白皮书整理）


笔者的个人理解
尽管扩散模型在二维（2D）数据合成领域已取得显著进展，

但如何运用扩散模型生成街景图像以实现图像增强，目前尚未得到理想的实现方案。现有的相关工作中，部分已通过二维边界框（2D bbox）和分割（segment）作为条件进行研究。然而，如何将条件拓展至相机位姿（embedding）、道路地图（embedding）、三维边界框（3D embedding）以及场景描述（文本）等作为控制条件，仍是有待探索的问题。这一拓展对于鸟瞰图（BEV）相关任务以及三维（3D）任务的发展具有重要意义。
MagicDrive 的主要思路是什么？
近期，扩散模型在与二维控制相关的数据合成方面取得了显著进展。然而，在街景生成中，精确的三维控制对于三维感知任务至关重要，目前却难以实现。将鸟瞰图（BEV）作为主要条件时，常常会在几何控制（如高度控制）方面面临挑战，进而影响目标形状、遮挡模式以及道路表面高程的准确表示。而这些因素对于感知数据合成，尤其是三维目标检测任务而言，是极为关键的。
MagicDrive 是一种创新的街景生成框架，它提供了多样化的三维几何控制手段，涵盖相机位姿、道路地图、三维边界框，并且通过定制化的编码策略实现了对文本描述的控制。此外，该框架还集成了一个跨视图注意力模块，能够确保在多个摄像机视图之间保持一致性。通过 MagicDrive，实现了高保真度的街景合成，能够捕捉到细微的三维几何特征以及各种场景描述，从而有效提升了诸如鸟瞰图分割（BEV 分割）和三维目标检测等任务的性能。
领域目前的工作
条件生成的扩散模型
扩散模型通过学习从高斯噪声分布到图像分布的渐进去噪过程来生成图像。由于这类模型在处理各种形式的控制以及多种条件时具有良好的适应性和能力，因此在文本到图像的合成、图像修复以及指导性图像编辑等多种任务中均表现优异。此外，从几何标注中合成的数据能够为下游任务提供帮助，例如二维目标检测。基于此，本文着重探讨了文本到图像（text-to-image，T2I）扩散模型在生成街景图像方面的潜力，以及其对下游三维感知模型的促进作用。
街景生成
许多现有的街景生成模型以二维布局作为条件，如二维边界框和语义分割。这些方法主要利用与图像比例直接对应的二维布局信息，而三维信息并不具备这种直接对应特性，这使得这些方法难以有效利用三维信息进行生成。在带有三维几何的街景合成研究中，BEVGen 是首个进行尝试的模型，它采用鸟瞰图（BEV）地图作为道路和车辆的生成条件。然而，由于其省略了高度信息，限制了它在三维目标检测中的应用。BEVControl 通过高度提升过程弥补了目标高度信息的损失，但从三维到二维的投影过程导致了关键三维几何信息（如深度和遮挡信息）的丢失。因此，这两种方法都未能充分利用三维标注信息，也无法实现对驾驶场景的文本控制。而 MagicDrive 提出分别对边界框和道路地图进行编码，以实现更为精细的控制，并整合场景描述信息，从而为街景生成提供了更强的控制能力。
三维场景的多摄像机图像生成
在三维场景的多摄像机图像生成中，视角一致性是基本要求。在室内场景的研究中，已有部分工作解决了这一问题。例如，MVDiffusion 利用全景图像和交叉视图注意力模块来保持全局一致性，而 pose-guided diffusion 则将极线几何作为约束先验。然而，这些方法主要依赖于图像视图的连续性，这在街景场景中并不总是能够满足，因为街景中的摄像机重叠范围有限。MagicDrive 在 UNet 中引入了额外的跨视图注意力模块，显著增强了多摄像机视图之间的一致性。
MagicDrive 的优势有哪些？
尽管 MagicDrive 框架结构简洁，但在生成与道路地图、三维边界框以及多样化摄像机视角相匹配的逼真图像方面表现卓越。此外，生成的图像能够有效增强三维目标检测和鸟瞰图分割（BEV 分割）任务的训练效果。MagicDrive 在场景、背景和前景层面均提供了全面的几何控制能力，这种灵活性使其能够生成前所未有的街景视图，适用于各种仿真需求。本文的主要贡献总结如下：
引入 MagicDrive 这一创新框架，能够生成基于鸟瞰图（BEV）以及为自动驾驶定制的三维数据的多透视摄像机视图。
开发了简单而高效的策略，成功应对了多摄像机视图一致性的挑战，并实现了对三维几何数据的有效管理。
通过严谨的实验验证，MagicDrive 在性能上超越了先前的街景生成技术，尤其是在多维度可控性方面表现突出。实验结果表明，使用 MagicDrive 合成的数据在三维感知任务中带来了显著的性能提升。




笔者的个人理解
虽然扩散模型在2D数据合成上已有很大进展，但如何使用扩散模型生成街景图像用于图像增强并没有很好的实现，当前已有工作通过2D bbox和segment作为条件，如何拓展到相机位姿（embedding），道路地图（embedding）和3D边界框（embedding）及场景描述（文本）作为控制条件呢？这将对BEV和3D任务有很大帮助。

MagicDrive主要思路是啥？
最近在扩散模型方面的进展显著提升了与2D控制相关的数据合成。然而,在街景生成中精确的3D控制对于3D感知任务至关重要,但仍然难以实现。将鸟瞰图（BEV）作为主要条件通常会导致在几何控制方面（例如高度）出现挑战,从而影响目标形状、遮挡模式和道路表面高程的表示,所有这些对于感知数据合成尤为重要,特别是对于3D目标检测任务而言。MagicDrive是一种新颖的街景生成框架,提供多样化的3D几何控制,包括相机位姿、道路地图和3D边界框,以及通过量身定制的编码策略实现的文本描述。此外还包括一个跨视图注意力模块,确保在多个摄像机视图之间保持一致性。通过MAGICDRIVE实现了高保真的街景合成,捕捉到微妙的3D几何和各种场景描述,从而增强了诸如BEV分割和3D目标检测等任务。

领域目前的工作
条件生成的扩散模型。 扩散模型通过学习从高斯噪声分布到图像分布的渐进去噪过程生成图像。由于它们在处理各种形式的控制和多种条件方面的适应性和能力,这些模型在各种任务中表现出色,如文本到图像的合成,修复以及指导性图像编辑。此外,从几何标注中合成的数据可以帮助下游任务,如2D目标检测。因此,本文探讨了text-to-image (T2I)扩散模型在生成街景图像并惠及下游3D感知模型方面的潜力。

街景生成。 许多街景生成模型以2D布局为条件,如2D边界框和语义分割。这些方法利用与图像比例直接对应的2D布局信息,而3D信息则不具备这种特性,因此使得这些方法不适用于利用3D信息进行生成。对于带有3D几何的街景合成,BEVGen是第一个进行尝试的。它使用BEV地图作为道路和车辆的条件。然而,省略高度信息限制了它在3D目标检测中的应用。BEVControl通过高度提升过程修正了目标高度的损失,但是从3D到2D的投影导致了关键的3D几何信息的丧失,如深度和遮挡。因此,它们都没有充分利用3D标注,也不能利用对驾驶场景的文本控制。MagicDrive提出分别对边界框和道路地图进行编码,以实现更为微妙的控制,并整合场景描述,提供对街景生成的增强控制。

3D场景的多摄像机图像生成 基本上需要视角一致性。在室内场景的背景下,一些研究已经解决了这个问题。例如,MVDiffusion使用全景图像和交叉视图注意力模块来保持全局一致性,而pose-guided diffusion则利用极线几何作为约束先验。然而,这些方法主要依赖于图像视图的连续性,而在街景中并不总是满足,因为摄像机重叠有限。MAGICDRIVE在UNet中引入了额外的跨视图注意力模块,显著增强了跨多摄像机视图的一致性。

MagicDrive的优势有哪些？
尽管MAGICDRIVE框架非常简单,但在生成与道路地图、3D边界框和多样化摄像机视角相一致的逼真图像方面表现出色。此外,生成的图像可以增强对3D目标检测和BEV分割任务的训练。MAGICDRIVE在场景、背景和前景层面提供了全面的几何控制。这种灵活性使其能够创造出以前未曾见过的适用于仿真目的的街景视图。总结本工作的主要贡献如下：

引入了MAGICDRIVE,这是一个创新的框架,生成基于BEV和为自动驾驶量身定制的3D数据的多透视摄像机视图。

开发了简单而强大的策略,有效应对多摄像机视图一致性的挑战,对3D几何数据进行管理。

通过严格的实验证明,MAGICDRIVE在先前的街景生成技术方面表现出色,尤其是在多维度可控性方面。此外结果显示,合成数据在3D感知任务中带来了显著的改进。





















terns

, as 

shown

 in F

igure

 2. C

onseq

uentl

y, ge

nerat

ing m

ulti-

camer

a str

eet-v

iew i

mages

 acco

rding



to 3D

 anno

tatio

ns be

comes

 vita

l to 

boost

 down

strea

m per

cepti

on ta

sks.



燕鸥，如图

所示 2。

因此，根据

 3D 注

释生成多相

机街景图像

对于促进下

游感知任务

变得至



关重要。



For s

treet

-view

 data

 synt

hesis

, two

 pivo

tal c

riter

ia ar

e rea

lism 

and c

ontro

llabi

lity.

 Real

ism r

equir

es



that 

the q

ualit

y of 

the s

ynthe

tic d

ata s

hould

 alig

n wit

h tha

t of 

real 

data;

 and 

in a 

given

 scen

e, vi

ews



from 

varyi

ng ca

mera 

persp

ectiv

es sh

ould 

remai

n con

siste

nt wi

th on

e ano

ther 

(Mild

enhal

l et 

al.,



2020)

. On 

the o

ther 

hand,

 cont

rolla

bilit

y emp

hasiz

es th

e pre

cisio

n in 

gener

ating

 stre

et-vi

ew



image

s tha

t adh

ere t

o pro

vided

 cond

ition

s: th

e BEV

 map,

 3D o

bject

 boun

ding 

boxes

, and

 came

ra



poses

 for 

views

. Bey

ond t

hese 

core 

requi

remen

ts, e

ffect

ive d

ata a

ugmen

tatio

n sho

uld a

lso g

rant



the f

lexib

ility

 to t

weak 

finer

 scen

ario 

attri

butes

, suc

h as 

preva

iling

 weat

her c

ondit

ions 

or th

e tim

e



of da

y. Ex

istin

g sol

ution

s lik

e BEV

Gen (

Swerd

low e

t al.

, 202

3) ap

proac

h str

eet v

iew g

enera

tion



by en

capsu

latin

g all

 sema

ntics

 with

in BE

V. Co

nvers

ely, 

BEVCo

ntrol

 (Yan

g et 

al., 

2023a

) sta

rts



by pr

oject

ing 3

D coo

rdina

tes t

o ima

ge vi

ews, 

subse

quent

ly us

ing 2

D geo

metri

c gui

dance

.



Howev

er, b

oth m

ethod

s com

promi

se ce

rtain

 geom

etric

 dime

nsion

s—hei

ght i

s los

t in 

BEVGe

n and



depth

 in B

EVCon

trol.



对于街景数

据合成，两

个关键的标

准是真实性

和可控性。

现实主义要

求合成数据

的质量应



与真实数据

的质量一致

；并且在给

定场景中，

来自不同摄

像机视角的

视图应该保

持彼此一



致(Mil

denha

ll et

 al.,

2020)

.另一方面

，可控性强

调生成街景

图像的精度

，这些图像

符合



提供的条件

:BEV 

图、3D 

对象边界框

和视图的相

机姿态。除

了这些核心

要求之外，

有效的数



据扩充还应

该能够灵活

地调整更精

细的场景属

性，例如主

要的天气条

件或一天中

的时间。



现有的解决

方案如 B

EVGen

(Swer

dlow 

et al

.,202

3)通过将

所有语义封

装在 BE

V 中来实

现街



景生成。相

反，BEV

Contr

ol(Ya

ng et

 al.,

2023a

)通过将 

3D 坐标

投影到图像

视图开始，

随



后使用 2

D 几何引

导。然而，

这两种方法

都牺牲了某

些几何尺寸

——bev

 gen 

中损失了高



度，BEV

Contr

ol 中损

失了深度。



The r

ise o

f dif

fusio

n mod

els h

as si

gnifi

cantl

y pus

hed t

he bo

undar

ies o

f con

troll

able 

image

 gene

ratio

n qua

lity.

 Spec

ifica

lly, 

Contr

olNet

 (Zha

ng et

 al.,

 2023

a) pr

opose

s a f

lexib

le fr

amewo

rk to

 inco

rpora

te 2D

 spat

ial c

ontro

ls ba

sed o

n pre

-trai

ned T

ext-t

o-Ima

ge (T

2I) d

iffus

ion m

odels



(Romb

ach e

t al.

, 202

2). H

oweve

r, 3D

 cond

ition

s are

 dist

inct 

from 

pixel

-leve

l con

ditio

ns or

 text

.



The c

halle

nge o

f sea

mless

ly in

tegra

ting 

them 

with 

multi

-came

ra vi

ew co

nsist

ency 

in st

reet 

view



synth

esis 

remai

ns.



扩散模型的

兴起极大地

拓展了可控

图像生成质

量的边界。

具体来说，

Contr

olNet

(Zhan

g et



al.,2

023a)

基于预先训

练的文本到

图像(T2

I)扩散模

型，提出了

一个灵活的

企业内部 

2D 空



间控制框架

(Romb

achet

 al.,

2022)

.但是，3

D 条件不

同于像素级

条件或文本

。将它们与

街



景合成中的

多摄像机视

图一致性无

缝集成的挑

战仍然存在

。



In th

is pa

per, 

we in

trodu

ce MA

GICDR

IVE, 

a nov

el fr

amewo

rk de

dicat

ed to

 stre

et-vi

ew sy

nthes

is



with 

diver

se 3D

 geom

etry 

contr

ols1



. For

 real

ism, 

we ha

rness

 the 

power

 of p

re-tr

ained

 stab

le di

ffusi

on (R

ombac

h et 

al., 

2022)

, fur

ther 

fine-

tunin

g it 

for s

treet

 view

 gene

ratio

n. On

e dis

tinct

ive



compo

nent 

of ou

r fra

mewor

k is 

the c

ross-

view 

atten

tion 

modul

e. Th

is si

mple 

yet e

ffect

ive c

om-



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR 

2024



5



ponen

t pro

vides

 mult

i-vie

w con

siste

ncy t

hroug

h int

eract

ions 

betwe

en ad

jacen

t vie

ws. I

n con

trast



to pr

eviou

s met

hods,

 MAGI

CDRIV

E pro

poses

 a se

parat

e des

ign f

or ob

jects

 and 

road 

map e

ncodi

ng



to im

prove

 cont

rolla

bilit

y wit

h 3D 

data.

 More

 spec

ifica

lly, 

given

 the 

seque

nce-l

ike, 

varia

blele

ngth 

natur

e of 

3D bo

undin

g box

es, w

e emp

loy c

ross-

atten

tion 

akin 

to te

xt em

beddi

ngs f

or th

eir



encod

- ing

. Bes

ides,

 we p

ropos

e tha

t an 

addic

tive 

encod

er br

anch 

like 

Contr

olNet

 (Zha

ng et

 al.,



2023a

) can

 enco

de ma

ps in

 BEV 

and i

s cap

able 

of vi

ew tr

ansfo

rmati

on. T

heref

ore, 

our d

esign



achie

ves g

eomet

ric c

ontro

ls wi

thout

 reso

rting

 to a

ny ex

plici

t geo

metri

c tra

nsfor

matio

ns or



impos

ing g

eomet

- ric

 cons

train

ts on

 mult

i-cam

era c

onsis

tency

. Fin

ally,

 MAGI

CDRIV

E fac

tors 

in



textu

al de

scrip

tions

, off

ering

 attr

ibute

 cont

rol s

uch a

s wea

ther 

condi

tions

 and 

time 

of da

y.



在本文中，

我们介绍了

 MAGI

CDRIV

E，这是一

个新颖的框

架，致力于

使用不同的

 3D 几

何控件



进行街景合

成 1。为

了逼真，我

们利用预先

训练的稳定

扩散(Ro

mbach

 et a

l.,20

22)，为

街



景生成进一

步微调。我

们的框架的

一个独特的

组成部分是

交叉视图注

意模块。这

个简单而



有 效 的

 组 件 

通 过 相

 邻 视 

图 之 间

 的 交 

互 提 供

 了 多 

视 图 一

 致 性 

。 与 以

 前 的 

方 法 相



比，MAG

ICDRI

VE 提出

了一种针对

对象和路线

图编码的独

立设计，以

提高 3D

 数据的可

控性。



更具体地说

，给定 3

D 边界框

的类似序列

的、可变长

度的性质，

我们使用类

似于文本嵌

入的



交叉注意进

行编码。此

外，我们建

议一个上瘾

的编码器分

支，如 C

ontro

lNet(

Zhang

 et



al.,2

023a)

可以在 B

EV 中对

地图进行编

码，并且能

够进行视图

变换。因此

，我们的设

计实



现了几何控

制，而无需

求助于任何

显式的几何

变换或对多

摄像机一致

性施加几何

约束。最



后，MAG

ICDRI

VE 在文

本描述中加

入了一些因

素，提供了

一些属性控

制，比如天

气状况和一



天中的时间

。



Our M

AGICD

RIVE 

frame

work,

 desp

ite i

ts si

mplic

ity, 

excel

s in 

gener

ating

 stri

kingl

y rea

listi

c



image

s & v

ideos

 that

 alig

n wit

h roa

d map

s, 3D

 boun

ding 

boxes

, and

 vari

ed ca

mera 

persp

ectiv

es.



Besid

es, t

he im

ages 

produ

ced c

an en

hance

 the 

train

ing f

or bo

th 3D

 obje

ct de

tecti

on an

d BEV



segme

ntati

on ta

sks. 

Furth

ermor

e, MA

GICDR

IVE o

ffers

 comp

rehen

sive 

geome

tric 

contr

ols a

t the



scene

, bac

k- gr

ound,

 and 

foreg

round

 leve

ls. T

his f

lexib

ility

 make

s it 

possi

ble t

o cra

ft pr

eviou

sly



unsee

n str

eet v

iews 

suita

ble f

or si

mulat

ion p

urpos

es. W

e sum

mariz

e the

 main

 cont

ribut

ions 

of th

is



work 

as:



我们的 M

AGICD

RIVE 

框架尽管简

单，但在生

成惊人逼真

的图像和视

频方面表现

出色，这些

图



像和视频与

道路地图、

3D 边界

框和各种相

机视角保持

一致。此外

，所产生的

图像可以增

强



3D 对象

检测和 B

EV 分割

任务的训练

。此外，M

AGICD

RIVE 

在场景、背

景和前景级

别提供全面



的几何控制

。这种灵活

性使之有可

能制作出以

前看不到的

适合于模拟

目的的街景

。我们将



这项工作的

主要贡献总

结为:



• The

 intr

oduct

ion o

f MAG

ICDRI

VE, a

n inn

ovati

ve fr

amewo

rk th

at ge

nerat

es mu

lti-p

erspe

ctive



camer

a vie

ws & 

video

s con

ditio

ned o

n BEV

 and 

3D da

ta ta

ilore

d for

 auto

nomou

s dri

ving.



• MAG

ICDRI

VE 的推

出，这是一

个创新的框

架，可以根

据 BEV

 和为自动

驾驶定制的

 3D 数

据生



成多视角摄

像机视图和

视频。



• The

 deve

lopme

nt of

 simp

le ye

t pot

ent s

trate

gies 

to ma

nage 

3D ge

ometr

ic da

ta, e

ffect

ively

 addr

essin

g the

 chal

lenge

s of 

multi

-came

ra vi

ew co

nsist

ency 

in st

reet 

view 

gener

ation

.



• 开发简

单而有效的

策略来管理

 3D 几

何数据，有

效解决街景

生成中多摄

像机视图一

致性的



挑战。



• Thr

ough 

rigor

ous e

xperi

ments

, we 

demon

strat

e tha

t MAG

ICDRI

VE ou

tperf

orms 

prior

 stre

et vi

ew



gener

ation

 tech

nique

s, no

tably

 for 

the m

ulti-

dimen

siona

l con

troll

abili

ty. A

dditi

onall

y, ou

r



resul

ts re

veal 

that 

synth

etic 

data 

deliv

ers c

onsid

erabl

e imp

rovem

ents 

in 3D

 perc

eptio

n tas

ks.



• 通过严

格的实验，

我们证明 

MAGIC

DRIVE

 优于现有

的街景生成

技术，特别

是在多维可

控性



方面。此外

，我们的结

果表明，合

成数据在 

3D 感知

任务中提供

了相当大的

改进。



1



In th

is pa

per, 

our 3

D geo

metry

 cont

rols 

conta

in co

ntrol

 from

 road

 maps

, 3D 

objec

t box

es, a

nd ca

mera



poses

. We 

do no

t con

sider

 othe

rs li

ke th

e exa

ct sh

ape o

f obj

ects 

or ba

ckgro

und c

onten

ts.



1 在本文

中，我们的

 3D 几

何控件包含

来自路线图

、3D 对

象框和相机

姿态的控件

。我们不认

为其他人喜

欢物体的确

切形状或背

景内容。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR 

2024



6



3 REL

ATED 

WORK



4 相关著

作



Diffu

sion 

Model

s for

 Cond

ition

al Ge

nerat

ion. 

Diffu

sion 

model

s (Ho

 et a

l., 2

020; 

Song 

et al

.,



2020;

 Zhen

g et 

al., 

2023)

 gene

rate 

image

s by 

learn

ing a

 prog

ressi

ve de

noisi

ng pr

ocess

 from

 the



Gauss

ian n

oise 

distr

ibuti

on to

 the 

image

 dist

ribut

ion. 

These

 mode

ls ha

ve pr

oven 

excep

tiona

l



acros

s div

erse 

tasks

, suc

h as 

text-

to-im

age s

ynthe

sis (

Romba

ch et

 al.,

 2022

; Nic

hol e

t al.

, 202

2;



Yang 

et al

., 20

23b),

 inpa

intin

g (Wa

ng et

 al.,

 2023

a), a

nd in

struc

tiona

l ima

ge ed

iting

 (Zha

ng et

 al.,



2023b

; Bro

oks e

t al.

, 202

3), d

ue to

 thei

r ada

ptabi

lity 

and c

ompet

ence 

in ma

nagin

g var

ious 

form 

of



contr

ols (

Zhang

 et a

l., 2

023a;

 Li e

t al.

, 202

3b) a

nd mu

ltipl

e con

ditio

ns (L

iu et

 al.,

 2022

a; Ga

o et



al., 

2023)

. Bes

ides,

 data

 synt

hesiz

ed fr

om ge

ometr

ic an

notat

ions 

can a

id do

wnstr

eam t

asks 

such 

as



2D ob

ject 

detec

- tio

n (Ch

en et

 al.,

 2023

c; Wu

 et a

l., 2

023b)

. Thu

s, th

is pa

per e

xplor

es th

e



poten

tial 

of T2

I dif

fusio

n mod

els i

n gen

erati

ng st

reet-

view 

image

s and

 bene

fitin

g dow

nstre

am 3D



perce

ption

 mode

ls.



条 件 生

 成 的 

扩 散 模

 型 。 

扩 散 模

 型 (H

o et 

al.,2

020 ；

 Song

 et a

l.,20

20 ； 

Zheng

 et



al.,2

0232)

通过学习从

高斯噪声分

布到图像分

布的渐进去

噪过程来生

成图像。这

些模型已



经被证明在

不同的任务

中表现出色

，例如文本

到图像的合

成(Rom

bach 

et al

.,202

2；



Nicho

l et 

al.,2

022；Y

ang e

t al.

,2023

b)，修复

(Wang

 et a

l.,20

23a)，

以及指导性

图像



编辑(Zh

ang e

t al.

,2023

b；Bro

okset

 al.,

2023)

，因为他们

在管理各种

形式控制(

Zhang

et



al.,2

023a；

Li et

 al.,

2023b

)和多个条

件(Liu

 et a

l.,20

22a；G

ao et

 al.,

2023)

.此外，



从几何注释

合成的数据

可以帮助下

游任务，例

如 2D 

对象检测(

Chen 

et al

.,202

3c；Wu

 et



al.,2

023b)

.因此，本

文探讨了 

T2I 扩

散模型在生

成街景图像

和有益于下

游 3D 

感知模型方



面的潜力。



Stree

t Vie

w Gen

erati

on. N

umero

us st

reet 

view 

gener

ation

 mode

ls co

nditi

on on

 2D l

ayout

s, su

ch



as 2D

 boun

ding 

boxes

 (Li 

et al

., 20

23b) 

and s

emant

ic se

gment

ation

 (Wan

g et 

al., 

2022)

. The

se



metho

ds le

verag

e 2D 

layou

t inf

ormat

ion c

orres

pondi

ng di

rectl

y to 

image

 scal

e, wh

ereas

 the 

3D



infor

matio

n doe

s not

 poss

ess t

his p

roper

ty, t

hereb

y ren

derin

g suc

h met

hods 

unsui

table

 for



lever

aging

 3D i

nfor-

 mati

on fo

r gen

erati

on. F

or st

reet 

view 

synth

esis 

with 

3D ge

ometr

y, BE

VGen



(Swer

dlow 

et al

., 20

23) i

s the

 firs

t to 

explo

re. I

t uti

lizes

 a BE

V map

 as a

 cond

ition

 for 

both 

roads



and v

ehicl

es. H

oweve

r, th

e omi

ssion

 of h

eight

 info

rmati

on li

mits 

its a

pplic

ation

 in 3

D obj

ect



detec

tion.

 BEVC

ontro

l (Ya

ng et

 al.,

 2023

a) am

ends 

the l

oss o

f obj

ect’s

 heig

ht by

 the 

heigh

t-lif

ting



proce

ss. S

imila

rly, 

Wang 

et al

. (20

23b) 

also 

proje

cts 3

D box

es to

 came

ra vi

ews t

o gui

de



gener

ation

. How

ever,

 the 

proje

ction

 from

 3D t

o 2D 

resul

ts in

 the 

loss 

of es

senti

al 3D

 geom

etric



infor

matio

n, li

ke de

pth a

nd oc

clusi

on. I

n thi

s pap

er, w

e pro

pose 

to en

code 

bound

ing b

oxes 

and



road 

maps 

separ

ately

 for 

more 

nuanc

ed co

ntrol

 and 

integ

rate 

scene

 desc

ripti

ons, 

offer

ing e

nhanc

ed



contr

ol ov

er th

e gen

erati

on of

 stre

et vi

ews.



街景生成。

许多街景生

成模型以 

2D 布局

为条件，例

如 2D 

边界框(L

i et 

al.,2

023b)

和语义



分割(Wa

ng et

 al.,

2022)

.这些方法

利用直接对

应于图像比

例的 2D

 布局信息

，而 3D

 信息不



具有这种属

性，从而使

得这些方法

不适合利用

 3D 信

息来生成。

对于具有 

3D 几何

形状的街



景合成，B

EVGen

(Swer

dlow 

et al

.,202

3)率先探

索。它利用

 BEV 

图作为道路

和车辆的条



件。然而，

高度信息的

遗漏限制了

它在三维物

体检测中的

应用。饮料

控制 (Y

anget



al.,2

023a)

通过高度提

升过程修正

对象高度的

损失。同样

的，Wan

g et 

al.（2

023b)

还将



3D 框投

影到相机视

图以指导生

成。然而，

从 3D 

到 2D 

的投影导致

基本 3D

 几何信息

的丢失，



如深度和遮

挡。在本文

中，我们建

议对边界框

和道路地图

分别进行编

码，以实现

更细致的



控制，并集

成场景描述

，从而增强

对街景生成

的控制。



Multi

-came

ra Im

age G

enera

tion 

of a 

3D sc

ene f

undam

ental

ly re

quire

s vie

wpoin

t con

siste

ncy.



Sever

al st

udies

 have

 addr

essed

 this

 issu

e wit

hin t

he co

ntext

 of i

ndoor

 scen

es. F

or in

stanc

e,



MVDif

- fus

ion (

Tang 

et al

., 20

23) e

mploy

s pan

orami

c ima

ges a

nd a 

cross

-view

 atte

ntion

 modu

le



to ma

intai

n glo

bal c

onsis

tency

, whi

le Ts

eng e

t al.

 (202

3) le

verag

e epi

polar

 geom

etry 

as a



const

raini

ng pr

ior. 

These

 appr

oache

s, ho

wever

, pri

maril

y rel

y on 

the c

ontin

uity 

of im

age v

iews,

 a



condi

tion 

not a

lways

 met 

in st

reet 

views

 due 

to li

mited

 came

ra ov

erlap

 and 

diffe

rent 

camer

a



confi

gurat

ions 

(e.g.

, exp

o- su

re, i

ntrin

sic).

 Our 

MAGIC

DRIVE

 intr

oduce

s ext

ra cr

oss-v

iew a

ttent

ion



modul

es to

 UNet

, whi

ch si

gnifi

cantl

y enh

ances

 cons

isten

cy ac

ross 

multi

-came

ra vi

ews.



3D 场景

的多相机图

像生成从根

本上要求视

点的一致性

。一些研究

已经在室内

场景的背景

下



解决了这个

问题。例如

，MVDi

f- fu

sion(

Tang 

et al

.,202

3)采用全

景图像和交

叉视图注意



模块来保持

全局一致性

，而 Ts

eng e

t al.

（2023

)利用核几

何作为约束

先验。然而

，这些



i



j



i



j



0



0



∈ ∈



∈ ∈



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR 

2024



7



方法主要依

赖于图像视

图的连续性

，由于有限

的相机重叠

和不同的相

机配置 (

例如，曝光



的、内在的

)，在街道

视图中并不

总是满足这

一条件。我

们的 MA

GICDR

IVE 为

 UNet

 引入了额



外的跨视图

注意模块，

这显著增强

了多摄像机

视图的一致

性。



5 PRE

LIMIN

ARY



6 初步的



Probl

em Fo

rmula

tion.

 In t

his p

aper,

 we c

onsid

er th

e coo

rdina

te of

 the 

LiDAR

 syst

em as

 the 

ego



car’s

 coor

dinat

e, an

d par

amete

rize 

all g

eomet

ric i

nform

ation

 acco

rding

 to i

t. Le

t S =

 {M,



B, L}

 be t

he de

scrip

tion 

of a 

drivi

ng sc

ene a

round

 the 

ego v

ehicl

e, wh

ere M

 ∈ {0

, 1}



w×h×c



is th

e bin

ary m

ap re

prese

nting

 a w 

× h m

eter 

road 

area 

in BE

V wit

h c s

emant

ic cl

asses

,



B = {

(ci, 

bi)}



N



问题公式化

。在本文中

，我们将激

光雷达系统

的坐标视为

自我车的坐

标，并根据

它来参数



化所有的几

何信息。设

 S = 

{M，B，

L}是自我

车辆周围的

驾驶场景的

描述，其中

 M ∈



{0，1}

w×h×c

 是表示 

BEV 中

具有 c 

个语义类的

 w×h 

米道路区域

的二进制地

图，B =



{(ci，

bi)}N



repre

sents

 the 

3D bo

undin

g box

 posi

tion 

(bi =

 {(xj

, yj,

 zj)}



8 ∈ R



8×3



) and

 clas

s (ci

 ∈



C)



表示三维边

界框的位置

(bi =

 {(xj

，yj，z

j)}8 

∈ R8×

3)和类(

ci ∈ 

C)



for e

ach o

bject

 in t

he sc

ene, 

and L

 is t

he te

xt de

scrib

ing a

dditi

onal 

infor

matio

n abo

ut th

e sce

ne



(e.g.

, wea

ther 

and t

ime o

f day

). Gi

ven a

 came

ra po

se P 

= [K,

 R, T

] (i.

e., i

ntrin

sics,

 rota

tion,



and t

ransl

ation

), th

e goa

l of 

stree

t-vie

w ima

ge ge

nerat

ion i

s to 

learn

 a ge

nerat

or G(

·) wh

ich



synth

esize

s



对于场景中

的每个对象

，L 是描

述关于场景

的附加信息

的文本(例

如，天气和

一天中的时



间)。给定

相机姿态 

P = [

K，R，T

](即，内

在的、旋转

和平移)，

街景图像生

成的目标是

学



习生成器 

G()，该

生成器 G

()合成



reali

stic 

image

s I ∈

 R



H×W ×

3



corre

spond

ing t

o the

 scen

e S a

nd ca

mera 

pose 

P as,

 I = 

G(S,



P, z)

, whe

re z 

∼ N (

0, 1)

 is a

 rand

om no

ise f

rom G

aussi

an di

strib

ution

.



对应于场景

 S 和相

机姿态 P

 的真实感

图像 I 

∈ RH×

W ×3 

为，I =

 G(S，

P，z)，

其中



z∞N(0

，1)是来

自高斯分布

的随机噪声

。



Condi

tiona

l Dif

fusio

n Mod

els. 

Diffu

sion 

model

s (Ho

 et a

l., 2

020; 

Song 

et al

., 20

20) g

enera

te



data 

(x0) 

by it

erati

vely 

denoi

sing 

a ran

dom G

aussi

an no

ise (

xT ) 

for T

 step

s. Ty

pical

ly, t

o lea

rn



the d

enois

ing p

roces

s, th

e net

work 

is tr

ained

 to p

redic

t the

 nois

e by 

minim

izing

 the 

mean-

squar

e



error

:



条件扩散模

型。扩散模

型(Ho 

et al

.,202

0；Son

g et 

al.,2

0202)

通过对 T

 步随机高

斯噪声



(xT)进

行迭代去噪

来生成数据

(x0)。

通常，为了

学习去噪过

程，网络被

训练成通过

最小化



均方误差来

预测噪声:



ℓsimp

le = 

Ex ,c

,ϵ,t 

||ϵ −

 ϵθ (



√



α¯tx0

 +



√



1 − α

¯t ϵ,

 t, c

)||2



, (1)



ℓ ϵ ϵ

ϵ ϵ s

imple

 = ex

 ,c, 

,t|| 

θ(√αt

x0+√1

αt ，t

，c)||

2，



(1)



where

 ϵθ i

s the

 netw

ork t

o tra

in, w

ith p

arame

ters 

θ, c 

is op

tiona

l con

ditio

ns, w

hich 

is us

ed fo

r the



condi

tiona

l gen

erati

on, t

[0, T

 ] is

 the 

time-

step,

 ϵ(0,

 I) i

s the

 addi

tive 

Gauss

ian n

oise,

 and



α¯ t 

is a 

scala

r par

amete

r. La

tent 

diffu

sion 

model

s (LD

M) (R

ombac

h et 

al., 

2022)

 is a

 spec

ial k

ind



of di

ffusi

on mo

del, 

where

 they

 util

ize a

 pre-

train

ed Ve

ctor 

Quant

ized 

Varia

tiona

l Aut

oEnco

der



(VQ其中

，ϵθ 是

要训练的网

络，参数为

 θ，c 

是可选条件

，用于条件

生成，t[

0，T ]

是时间



步长，ϵ(

0，I)是

加性高斯噪

声，α t

 是标量参

数。潜在扩

散模型(L

DM)(R

ombac

h et



al.,2

022)是

一种特殊的

扩散模型，

其中他们利

用预训练的

矢量量化变

分自动编码

器(VQ



VAE) 

(Esse

r et 

al., 

2021)

 and 

perfo

rm di

ffusi

on pr

ocess

 in t

he la

√



tent 

space

. G√



iven 

the V

Q-VAE



VAE)(

Esser

 et a

l.,20

21)并在

 la√t

ent 空

间进行扩散

处理。g√

艾文 VQ

-VAE



encod

er as

 z = 

E(x),

 one 

can r

ewrit

e ϵθ(

·) in

 Equa

tion 

1 as 

ϵθ( α

¯tE(x

0) + 

1 −



α¯t ϵ

, t, 

c) fo

r



编码器为 

z = E

(x)，可

以将 ϵθ

()改写为

等式 1 

作为 ϵθ

( α t

E(x0)

 + 1α

tϵ，t，

c)



LDM. 

Besid

es, L

DM co

nside

rs te

xt de

scrib

ing t

he im

age a

s con

ditio

n c.



LDM。此

外，LDM

 认为描述

图像的文字

是条件 c



Promp

t:



“A dr

iving

 scen

e <de

scrip

tion>

”



UNet 

“A dr

iving

 scen

e ...

 ”



QQ



KVKV



Camer

a Pos

e



Objec

t Box



“A dr

iving

 scen

e ...

 Rain

 ...”



Seq E

mbedd

ing:



<CAM>

 <TXT

> <BO

X 1> 

... <

BOX n

>



Road 

Map Q

 KV C

ross-

view 

Atten

tionT

raina

ble C

ross-

atten

tion 

injec

tion 

Froze

n Add

ictiv

e enc

oder 

injec

Promp

t:



“A dr

iving

 scen

e <de

scrip

tion>

”



UNet 

“A dr

iving

 scen

e ...

 ”



QQ



KVKV



Camer

a Pos

e



Objec

t Box



“A dr

iving

 scen

e ...

 Rain

 ...”



Seq E

mbedd

ing:



<CAM>

 <TXT

> <BO

X 1> 

... <

BOX n

>



Road 

Map Q

 KV C

ross-

view 

Atten

tionT

raina

ble C

ross-

atten

tion 

injec

tion 

Froze

n Add

ictiP

ublis

hed a

s a c

onfer

ence 

paper

 at I

CLR 2

024



8



Figur

e 3: 

Overv

iew o

f MAG

ICDRI

VE fo

r str

eet-v

iew i

mage 

gener

ation

. MAG

ICDRI

VE ge

nerat

es



highl

y rea

listi

c ima

ges, 

explo

iting

 geom

etric

 info

rmati

on fr

om 3D

 anno

tatio

ns by

 inde

pende

ntly



encod

ing r

oad m

aps, 

objec

t box

es, a

nd ca

mera 

param

eters

 for 

preci

se, g

eomet

ry-gu

ided 

synth

esis.



Addit

ional

ly, M

AGICD

RIVE 

accom

modat

es gu

idanc

e fro

m des

cript

ive c

ondit

ions 

(e.g.

, wea

ther)

.



图 3:用

于街景图像

生成的 M

AGICD

RIVE 

概述。MA

GICDR

IVE 生

成高度逼真

的图像，通

过独立



编码路线图

、对象框和

相机参数，

利用 3D

 注释中的

几何信息进

行精确的几

何导向合成

。此



外，MAG

ICDRI

VE 还支

持来自描述

性条件(如

天气)的指

导。



7 STR

EET V

IEW G

ENERA

TION 

WITH 

3D IN

FORMA

TION



8 基于 

3D 信息

的街景生成



The o

vervi

ew of

 MAGI

CDRIV

E is 

depic

ted i

n Fig

ure 3

. Ope

ratin

g on 

the L

DM pi

pelin

e, MA

GICDR

IVE g

enera

tes s

treet

-view

 imag

es co

nditi

oned 

on bo

th sc

ene a

nnota

tions

 (S) 

and t

he ca

mera



pose 

(P) f

or ea

ch vi

ew. G

iven 

the 3

D geo

metri

c inf

ormat

ion i

n sce

ne an

notat

ions,

 proj

ectin

g all

 to



a BEV

 map,

 akin

 to B

EVGen

 (Swe

rdlow

 et a

l., 2

023) 

or BE

VCont

rol (

Yang 

et al

., 20

23a),

 does

n’t



ensur

e pre

cise 

guida

nce f

or st

reet 

view 

gener

ation

, as 

exemp

lifie

d in 

Figur

e 2. 

Conse

quent

ly,



MAG- 

ICDRI

VE ca

tegor

izes 

condi

tions

 into

 thre

e lev

els: 

scene

 (tex

t and

 came

ra po

se), 

foreg

round



(3D b

oundi

ng bo

xes),

 and 

backg

round

 (roa

d map

); an

d int

egrat

es th

em se

parat

ely v

ia cr

ossat

tenti

on an

d an 

addit

ive e

ncode

r bra

nch, 

detai

led i

n Sec

tion 

4.1. 

Addit

ional

ly, m

ainta

ining



consi

stenc

y acr

oss d

iffer

ent c

amera

s is 

cruci

al fo

r syn

thesi

zing 

stree

t vie

ws. T

hus, 

we in

trodu

ce a



simpl

e yet

 effe

ctive

 cros

s-vie

w att

entio

n mod

ule i

n Sec

tion 

4.2. 

Lastl

y, we

 eluc

idate

 our 

train

ing



strat

egies

 in S

ectio

n 4.3

, emp

hasiz

ing C

lassi

fier-

Free 

Guida

nce (

CFG) 

in in

tegra

ting 

vario

us



condi

tions

.



MAGIC

DRIVE

 的概述如

图所示 3

。MAG-

 ICDR

IVE 在

 LDM 

管道上运行

，根据每个

视图的场景

注



释和相机姿

态生成街景

图像。给定

场景注释中

的 3D 

几何信息，

将所有投影

到 BEV

 图，类似



于 BEV

Gen(S

werdl

ow et

 al.,

2023)

或 BEV

Contr

ol(Ya

ng et

 al.,

2023a

)，不能确

保街景生



成的精确指

导，如图所

示 2。因

此，MAG

- ICD

RIVE 

将条件分为

三个级别:

场景(文本

和相机



姿势)、前

景(3D 

边界框)和

背景(路线

图)；并通

过交叉注意

和附加编码

器分支分别

集成它



们，详见第

节 4.1

。此外，保

持不同摄像

机之间的一

致性对于合

成街景至关

重要。因此

，



我们在第一

节介绍了一

个简单而有

效的交叉视

角注意模块

 4.2。

最后，我们

在第二节阐

述



了我们的培

训策略 4

.3，强调

无分类器引

导(CFG

)在综合各

种条件。



8.1 G

EOMET

RIC C

ONDIT

IONS 

ENCOD

ING



8.2 几

何条件编码



{ } {

 }



{ ∈ ∈

 ∈ }



{ } {

 }



{ ∈ ∈

 ∈ }



∈



∈



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR 

2024



9



As il

lustr

ated 

in Fi

gure 

3, tw

o str

ategi

es ar

e emp

loyed

 for 

infor

matio

n inj

ectio

n int

o the

 UNet

 of



diffu

sion 

model

s: cr

oss-a

ttent

ion a

nd ad

ditiv

e enc

oder 

branc

h. Gi

ven t

hat t

he at

tenti

on me

chani

sm (V

aswan

i et 

al., 

2017)

 is t

ailor

ed fo

r seq

uenti

al da

ta, c

ross-

atten

tion 

is ap

t for

 mana

ging 

varia

ble l

ength

 inpu

ts li

ke te

xt to

kens 

and b

oundi

ng bo

xes. 

Conve

rsely

, for

 grid

-like

 data

, suc

h as 

road



maps,

 the 

addit

ive e

ncode

r bra

nch i

s eff

ectiv

e in 

infor

matio

n inj

ectio

n (Zh

ang e

t al.

, 202

3a).



There

- for

e, MA

GICDR

IVE e

mploy

s dis

tinct

 enco

ding 

modul

es fo

r var

ious 

condi

tions

.



如图所示 

3 在扩散

模型的 U

Net 中

，信息注入

采用了两种

策略:交叉

注意和附加

编码分支。



假设注意力

机制(Va

swani

 et a

l.,20

17)是为

顺序数据定

制的，所以

交叉注意适

用于管理可



变长度的输

入，如文本

标记和边界

框。相反，

对于像道路

地图这样的

网格状数据

，加法编



码器分支在

信息注入(

Zhang

 et a

l.,20

23a).

因此，MA

GICDR

IVE 针

对各种情况

采用不同的



编码模块。



Scene

-leve

l Enc

oding

 incl

udes 

camer

a pos

e P =

 K R



3×3



, R R



3×3



, T R



3×1



, and

 text



seque

nce L

. For

 text

, we 

const

ruct 

the p

rompt

 with

 a te

mplat

e as 

“A dr

iving

 scen

e ima

ge



at lo

catio

n . d

escri

ption

 ”, a

nd le

verag

e a p

re-tr

ained

 CLIP

 text

 enco

der



(Etex

t) as

 LDM 

(Romb

ach e

t al.

, 202

2), a

s sho

wn by

 Equa

tion 

2, wh

ere L

 is t

he to

ken l

ength

 of L

.



As fo

r



场景级编码

包括相机姿

态 P =

 K R3

×3，R 

R3×3，

T R3×

1 和文本

序列 l。

对于文本，



我们使用模

板将提示构

造为“某位

置的驾驶场

景图像”。

描述”，并

利用预训练

的剪



辑文本编码

器(Ete

xt)作为

 LDM(

Romba

ch et

 al.,

2022)

，如等式所

示 2，其

中 L 是

 L 的令



牌长度。至

于



camer

a pos

e, we

 firs

t con

cat e

ach p

arame

ter b

y the

ir co

lumn,

 resu

lting

 in P

¯ = [

K, R,

 T]



T



R



7×3



. Sin

ce P¯

 cont

ains 

value

s fro

m sin

/cos 

funct

ions 

and a

lso 3

D off

sets,

 to h

ave t

he



model

 effe

ctive

ly 相机

姿态，我们

首先按列连

接每个参数

，得到 P

 = [K

，R，T]

T R7×

3。因为 

P 包含来

 自正弦/

余弦函数的

值以及 3

D 偏移，

所以有效地

具有模型



inter

pret 

these

 high

-freq

uency

 vari

ation

s, we

 appl

y Fou

rier 

embed

ding 

(Mild

enhal

l et 

al., 

2020)

 to



为了解释这

些高频变化

，我们应用

傅立叶嵌入

(Mild

enhal

l et 

al.,2

020)到



each 

3-dim

 vect

or be

fore 

lever

aging

 a Mu

lti-L

ayer 

Perce

ption

 (MLP

, Eca

m) to

 embe

d the

 came

ra



pose 

param

eters

, as 

in Eq

uatio

n 3. 

To ma

intai

n con

siste

ncy, 

we se

t the

 dime

nsion

 of h



c



the s

ame



as th

at of

 h



t



. Thr

ough 

the C

LIP t

ext e

ncode

r, ea

ch te

xt em

beddi

ng h



t



alrea

dy co

ntain

s pos

ition

al



在利用多层

感知(ML

P，Eca

m)来嵌入

相机姿态参

数之前的每

个三维向量

，如等式中

所示 3。



为了保持一

致性，我们

将 hc 

的维度设置

为与 ht

 的维度相

同。通过剪

辑文本编码

器，每个



文本嵌入 

hti我已

经包含位置

 i我



infor

matio

n (Ra

dford

 et a

l., 2

021).

 Ther

efore

, we 

prepe

nd th

e cam

era p

ose e

mbedd

ing h



c



to te

xt



embed

dings

, res

ultin

g in 

scene

-leve

l emb

eddin

g h



s = [

h



c



, h



t



].



信息(Ra

dford

 et a

l.,20

21).因

此，我们将

嵌入 hc

 的摄像机

姿态添加到

文本嵌入中

，从而得到



场景级嵌入

 hs =

 [hc，

ht]。



h



t = [

h



t



. . .

 h



t



] = E

text(

L), (

2)



ht = 

[ht。。

。ht ]

 = Et

ext(L

)， (2

)



1 L



一 L



h



c = E

cam(F

ourie

r(P¯ 

)) = 

Ecam(

Fouri

er([K

, R, 

T]



T



)). (

3)



hc = 

Ecam(

傅立叶(P

 )) =

 Ecam

(傅立叶(

[K，R，

T]T))

。 (3)



3D Bo

undin

g Box

 Enco

ding.

 Sinc

e eac

h dri

ving 

scene

 has 

a var

iable

 leng

th of

 boun

ding 

boxes

,



we in

ject 

them 

throu

gh th

e cro

ss-at

tenti

on me

chani

sm si

milar

 to s

cene-

level

 info

rmati

on.



Speci

fical

ly, w

e enc

ode e

ach b

ox in

to a 

hidde

n vec

tor h



b



, whi

ch ha

s the

 same

 dime

nsion

s as 

that



of h



t



. Eac

h 3D 

bound

ing b

ox (c

i, bi

) con

tains

 two 

types

 of i

nform

ation

: cla

ss la

bel c

i and

 box



posit

ion b

i. Fo

r



3D 边界

框编码。由

于每个驾驶

场景都有可

变长度的包

围盒，我们

通过类似于

场景级信息

的



交叉注意机

制来注入它

们。具体来

说，我们将

每个盒子编

码成一个隐

藏向量 h

b，它的维

数



与 ht 

的维数相同

。每个 3

D 边界框

(ci，b

i)包含两

种类型的信

息:类别标

签 ci 

和框位置



bi。为



Train

able 

Froze

n



self-

attn



<CAM>

 <TXT

> <BO

Xes> 

cross

-attn



Left 

Neigh

bor



cross

-view

 attn



Right

 Neig

hbor



zero-

init



Train

able 

Froze

n



self-

attn



<CAM>

 <TXT

> <BO

Xes> 

cross

-attn



Left 

Neigh

bor



cross

-view

 attn



Right

 Neig

hbor



zero-

init



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



Tansf

ormer

 Bloc

k



变压器组



witho

ut cr

oss-v

iew a

ttent

ion



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



i ∈



i ∈



v i i

 i



v i i

 i



没有交叉视

角的关注



with 

cross

-view

 atte

ntion



注意交叉视

角



Figur

e 4: 

Cross

-view

 Atte

ntion

. lef

t: we

 intr

oduce

 cros

s-vie

w att

entio

n to 

the p

re-tr

ained

 UNet



after

 the 

cross

-atte

ntion

 modu

le. r

ight:

 we h

ighli

ght s

ome a

reas 

for c

ompar

ison 

betwe

en wi

thout



and w

ith c

ross-

view 

atten

tion.

 Cros

s-vie

w att

entio

n gua

rante

es co

nsist

ency 

acros

s mul

tiple

 view

s.



图 4:交

叉视角注意

力。左图:

在交叉注意

力模块之后

，我们将交

叉注意力引

入预训练的



UNet。

右图:我们

突出显示了

一些区域，

以便在没有

交叉视图注

意力和有交

叉视图注意

力的



情况下进行

比较。跨视

图注意力保

证了多个视

图的一致性

。



class

 labe

ls, w

e uti

lize 

the m

ethod

 simi

lar t

o Li 

et al

. (20

23b),

 wher

e the

 pool

ed em

beddi

ngs o

f



class

 name

s (Lc

 ) ar

e con

sider

ed as

 labe

l emb

eddin

gs. F

or bo

x pos

ition

s bi 

R



8×3



,



repre

sente

d by 

the c

oordi

nates

 of i

ts 8 

corne

r poi

nts, 

we ut

ilize

 Four

ier e

mbedd

ing t

o eac

h poi

nt



and p

ass t

hroug

h an 

MLP f

or en

codin

g, as

 in E

quati

on 4.

 Then

, we 

use a

n MLP

 to c

ompre

ss



both 

class

 and 

posit

ion e

mbedd

ing i

nto o

ne hi

dden 

vecto

r, as

 in E

quati

on 5.

 The 

final

 hidd

en



state

s for

 all 

bound

ing b

oxes



类标签，我

们利用类似

于 Li 

et al

.（202

3b)，其

中类名(L

c)的池化

嵌入被认为

是标签嵌



入。对于由

 8 个角

点的坐标表

示的框位置

 bi R

8×3，我

们利用傅立

叶嵌入到每

个点，并通



过 MLP

 进行编码

，如等式所

示 4。然

后，我们使

用 MLP

 将类别和

位置嵌入压

缩到一个隐



藏向量中，

如等式所示

 5。所有

边界框的最

终隐藏状态

 of e

ach s

cene 

are r

epres

ented

 as h



b = [

h



b



. . .

 h



b



], wh

ere N

 is t

he nu

mber 

of bo

xes. 

表示为 h

b = [

hb。。。

hb ]，

其中 N 

是盒子的数

量。 1 

N



一 普通



e



b



(i) =

 AvgP

ool(E

text(

Lc ))

, e



b



(i) =

 MLPp

(Four

ier(b

i)), 

(4) e

b(i) 

= Avg

Pool(

Etext

(Lc))

，eb (

i) = 

MLPp(

Fouri

er(bi

))，(4

)



c i p



c 我 p



h



b = E

box(c

i, bi

) = M

LPb(e



b



(i), 

e



b



(i)).

 (5)



hb = 

Ebox(

ci，bi

) = M

LPb(e

b(i)，

eb(i)

)。 (5

)



i c p



我 c p



Ideal

ly, t

he mo

del l

earns

 the 

geome

tric 

relat

ionsh

ip be

tween

 boun

ding 

boxes

 and 

camer

a pos

e



throu

gh tr

ainin

g. Ho

wever

, the

 dist

ribut

ion o

f the

 numb

er of

 visi

ble b

oxes 

to di

ffere

nt vi

ews



is lo

ng-ta

iled.

 Thus

, we 

boots

trap 

learn

ing b

y fil

terin

g vis

ible 

objec

ts to

 each

 view

 (vi)

, i.e

.,



fviz 

in Eq

uatio

n 6. 

Besid

es, w

e als

o add

 invi

sible

 boxe

s for

 augm

entat

ion (

more 

detai

ls in

 Sect

ion



4.3).



理想情况下

，模型通过

训练学习边

界框和相机

姿态之间的

几何关系。

然而，不同

视图



的可见框的

数量分布是

长尾的。因

此，我们通

过将可见对

象过滤到每

个视图(v

i)，即



方程中的 

fviz，

来引导学习

 6。除此

之外，我们

还增加了隐

形的增强框

(更多细节

请见



4.3).



h



b= {h



b ∈ h



b



|fviz

(bi, 

Rv , 

Tv ) 

> 0}.

 (6)



hb= {

hb ∈ 

hb|fv

iz(bi

，Rv，T

v ) >

 0}。 

(6)



Road 

Map E

ncodi

ng. T

he ro

ad ma

p has

 a 2D

-grid

 form

at. W

hile 

Zhang

 et a

l. (2

023a)

 show

s



the a

ddict

ive e

ncode

r can

 inco

rpora

te th

is ki

nd of

 data

 for 

2D gu

idanc

e, th

e inh

erent

 pers

pecti

ve



diffe

rence

s bet

ween 

the r

oad m

ap’s 

BEV a

nd th

e cam

era’s

 Firs

t-Per

son V

iew (

FPV) 

creat

e



discr

ep- a

ncies

. BEV

Contr

ol (Y

ang e

t al.

, 202

3a) e

mploy

s a b

ack-p

rojec

tion 

to tr

ansfo

rm fr

om



BEV t

o FPV

 but 

compl

icate

s the

 situ

ation

 with

 an i

ll-po

sed p

roble

m. In

 MAGI

CDRIV

E, we

 prop

ose



that 

expli

cit v

iew t

ransf

ormat

ion i

s unn

ecess

ary, 

as su

ffici

ent 3

D cue

s (e.

g., h

eight

 from

 obje

ct



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



c



c



boxes

 and 

cam- 

era p

ose) 

allow

 the 

addic

tive 

encod

er to

 acco

mplis

h vie

w tra

nsfor

matio

n.



Speci

fical

ly, w

e int

egrat

e sce

ne-le

vel a

nd 3D

 boun

ding 

box e

mbedd

ings 

into 

the m

ap en

coder

 (see



Figur

e 3).

 Scen

e-lev

el em

- bed

dings

 prov

ide c

amera

 pose

s, an

d box

 embe

dding

s off

er ro

ad



eleva

tion 

cues.

 Addi

tiona

lly, 

incor

- por

ating

 text

 desc

ripti

ons f

acili

tates

 the 

gener

ation

 of r

oads



under

 vary

ing c

ondit

ions 

(e.g.

, wea

ther 

and t

ime o

f day

). Th

us, t

he ma

p enc

oder 

can s

ynerg

ize



with 

other

 cond

ition

s for

 gene

ratio

n.



路线图编码

。该路线图

具有 2D

 网格格式

。在…期间

 Zhan

g et 

al.（2

023a)

显示了令人

上瘾



的编码器可

以将这种数

据用于 2

D 制导，

路线图的 

BEV 和

相机的第一

人称视角(

FPV)之

间的



固有视角差

异造成了差

异。饮料控

制(Yan

g et 

al.,2

023a)

使用反向投

影从 BE

V 转换到



FPV，但

是用不适定

的问题使情

况变得复杂

。在 MA

GICDR

IVE 中

，我们提出

显式视图转

换是



不必要的，

因为足够的

 3D 线

索(例如，

距离对象框

的高度和 

cam 时

代的姿势)

允许附加编

码



器完成视图

转换。具体

来说，我们

将场景级和

 3D 边

界框嵌入集

成到地图编

码器中(见

图 3).



场景级嵌入

提供相机姿

态，而盒子

嵌入提供道

路高程线索

。此外，结

合文本描述

有助于在



不同条件下

(例如，天

气和一天中

的时间)生

成道路。因

此，map

 编码器可

以与用于生

成的



其他条件协

同。



8.3 C

ROSS-

VIEW 

ATTEN

TION 

MODUL

E



8.4 交

叉视野注意

模块



In mu

lti-c

amera

 view

 gene

ratio

n, it

 is c

rucia

l tha

t ima

ge sy

nthes

is re

mains

 cons

isten

t acr

oss



diffe

rent 

persp

ectiv

es. T

o mai

ntain

 cons

isten

cy, w

e int

roduc

e a c

ross-

view 

atten

tion 

modul

e (Fi

gure



4). G

iven 

the s

parse

 arra

ngeme

nt of

 came

ras i

n dri

ving 

conte

xts, 

each 

cross

-view

 atte

ntion

 allo

ws



the t

arget

 view

 to a

ccess

 info

rmati

on fr

om it

s imm

ediat

e lef

t and

 righ

t vie

ws, a

s in 

Equat

ion 7

;



here,

 t, l

, and

 r ar

e the

 targ

et, l

eft, 

and r

ight 

view 

respe

ctive

ly. T

hen, 

the t

arget

 view

 aggr

egate

s



such 

infor

matio

n wit

h ski

p con

necti

on, a

s in 

Equat

ion 8

, whe

re h



v



indic

ates 

the h

idden

 stat

e of 

the



targe

t vie

w.



在多相机视

图生成中，

图像合成在

不同视角之

间保持一致

是至关重要

的。为了保

持一致性，



我们引入了

交叉视图注

意模块(图

 4).给

定驾驶环境

中摄像机的

稀疏排列，

每个跨视图

注意



力允许目标

视图从其紧

邻的左视图

和右视图访

问信息，如

等式所示 

7；这里，

t、l 和

 r 分别



是目标视图

、左视图和

右视图。然

后，目标视

图通过跳过

连接聚集这

些信息，如

等式所示



8，其中 

hv 表示

目标视图的

隐藏状态。



Atten

tioni



(Qt, 

Ki, V

i) =



softm

ax(



Atten

tioni

 (Qt，

Ki，Vi

) =



softm

ax(



QtK



T



=



=



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



√



i



) · V

 , i 

∈ {l,

 r},



c



√ i )

 · V 

, i ∈

 {l, 

r},



QtKT



i



我



d



d



v



ou



t



v



out



v +



Atten

tionl



v +注意

l + A

ttent

ionr



h



h



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



i c



i c



+注意r 

. (8)

 。 (8

)



We in

ject 

cross

-view

 atte

ntion

 afte

r the

 cros

s-att

entio

n mod

ule i

n the

 UNet

 and 

apply

 zero

initi

aliza

tion 

(Zhan

g et 

al., 

2023a

) to 

boots

trap 

the o

ptimi

zatio

n. Th

e eff

icacy

 of t

he cr

oss-v

iew



atten

tion 

modul

e is 

demon

strat

ed in

 Figu

re 4 

right

, Fig

ure 5

, and

 Figu

re 6.

 The 

multi

layer

ed st

ructu

re of

 UNet

 enab

les a

ggreg

ating

 info

rmati

on fr

om lo

ng-ra

nge v

iews 

after

 seve

ral s

tacke

d blo

cks.



There

fore,

 usin

g cro

ss-vi

ew at

tenti

on on

 adja

cent 

views

 is e

nough

 for 

multi

-view

 cons

isten

cy, f

urthe

r evi

dence

d by 

the a

blati

on st

udy i

n App

endix

 C.



我们在 U

Net 中

的交叉注意

模块之后注

入交叉视图

注意，并应

用零初始化

(Zhan

g et



al.,2

023a)

来引导优化

。交叉视图

注意模块的

功效如图所

示 4 对

，数字 5

，和图 6

。UNet

 的



多层结构使

得能够在几

个堆叠的块

之后从远程

视图中聚集

信息。因此

，在相邻视

图上使用



跨视图注意

力对于多视

图一致性来

说是足够的

，附录中的

消融研究进

一步证明了

这一点



C。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



8.5 M

ODEL 

TRAIN

ING



8.6 模

特培训



Class

ifier

-free

 Guid

ance 

reinf

orces

 the 

impac

t of 

condi

tiona

l gui

dance

 (Ho 

& Sal

imans

, 202

1;



Romba

ch et

 al.,

 2022

). Fo

r eff

ectiv

e CFG

, mod

els n

eed t

o dis

card 

condi

tions

 duri

ng tr

ainin

g occ

asion

ally.

 Give

n the

 uniq

ue na

ture 

of ea

ch co

nditi

on, a

pplyi

ng a 

drop 

strat

egy i

s com

plex 

for



multi

ple c

ondit

ions.

 Ther

efore

, our

 MAGI

CDRIV

E sim

plifi

es th

is fo

r fou

r con

ditio

ns by



concu

rrent

ly dr

op- p

ing s

cene-

level

 cond

ition

s (ca

mera 

pose 

and t

ext e

mbedd

ings)

 at a

 rate

 of γ



s



.



For b

oxes 

and m

aps,



无分类器指

导加强了条

件指导的影

响(Ho 

& Sal

imans

,2021

；Romb

ach e

t al.

,2022

).对于



有效的 C

FG，模型

需要偶尔在

训练期间丢

弃条件。考

虑到每个条

件的独特性

质，对多个

条



件应用丢弃

策略是复杂

的。因此，

我们的 M

AGICD

RIVE 

通过以 γ

s 的速率

同时删除场

景级条



件(相机姿

态和文本嵌

入)来简化

这四种条件

。对于盒子

和地图，



which

 have

 sema

ntic 

repre

senta

tions

 for 

null 

(i.e.

, pad

ding 

token

 in b

oxes 

and 0

 in m

aps) 

in th

eir



encod

ing, 

we ma

intai

n the

m thr

ougho

ut tr

ainin

g. At

 infe

rence

, we 

utili

ze nu

ll fo

r all

 cond

ition

s,



enabl

ing m

eanin

gful 

ampli

ficat

ion t

o gui

de ge

nerat

ion.



在它们的编

码中具有 

null 

的语义表示

(即，在方

框中填充标

记，在映射

中填充 0

 ),我们

在



整个训练中

保持它们。

在推论中，

我们对所有

条件都使用

 null

，使得有意

义的扩增能

够指



导生成。



Train

ing O

bject

ive a

nd Au

gment

ation

. Wit

h all

 the 

condi

tions

 inje

cted 

as in

puts,

 we a

dapt 

the



train

ing o

bject

ive d

escri

bed i

n Sec

tion 

3 to 

the m

ulti-

condi

tion 

scena

rio, 

as in

 Equa

tion 

9.



培训目标和

强化。将所

有注入的条

件作为输入

，我们调整

第节中描述

的培训目标

 3 多条

件



场景，如等

式所示 9

。



ℓ = E

x0 ,ϵ

,t,{S

,P } 

||ϵ −

 ϵθ (



√



α¯tE(

x0) +



√



1 − α

¯t ϵ,

 t, {

S, P}

)|| .



(9)



ℓ ϵ ϵ

ϵ ϵ =

 ex0 

, ,t,

{s,p 

}|| θ

(√αte

(x0)+

√1αt 

，t，{S

，P})|

|。



(9)



Besid

es, w

e emp

hasiz

e two

 esse

ntial

 stra

tegie

s whe

n tra

ining

 our 

MAGIC

DRIVE

. Fir

st, t

o



count

eract

 our 

filte

ring 

of vi

sible

 boxe

s, we

 rand

omly 

add 1

0% in

visib

le bo

xes a

s an



augme

ntati

on, e

nhanc

ing t

he mo

del’s

 geom

etric

 tran

sform

ation

 capa

bilit

ies. 

Secon

d, to

 leve

rage



cross

-view

 atte

ntion

, whi

ch fa

cilit

ates 

infor

matio

n sha

ring 

acros

s mul

tiple

 view

s, we

 appl

y uni

que



noise

s to 

diffe

rent 

views

 in e

ach t

raini

ng st

ep, p

reven

ting 

trivi

al so

lutio

ns to

 Equa

tion 

9 (e.

g.,



outpu

tting

 the 

share

d com

ponen

t acr

oss d

iffer

ent v

iews)

. Ide

ntica

l ran

dom n

oise 

is re

serve

d



exclu

sivel

y for

 infe

rence

.



此外，我们

在训练 M

AGICD

RIVE 

时强调两个

基本策略。

首先，为了

抵消我们对

可见框的过



滤，我们随

机添加了 

10%的不

可见框作为

增强，增强

了模型的几

何变换能力

。第二，为

了



利用跨视图

注意力，这

有助于跨多

个视图的信

息共享，我

们在每个训

练步骤中对

不同的视



图应用独特

的噪声，防

止方程的平

凡解 9(

例如，跨不

同视图输出

共享组件)

。完全相同

的随



机噪声专门

用于推断。



9 EXP

ERIME

NTS



10 实验



10.1 

EXPER

IMENT

AL SE

TUPS



10.2 

实验装置



Datas

et an

d Bas

eline

s. We

 empl

oy th

e nuS

cenes

 data

set (

Caesa

r et 

al., 

2020)

, a p

reval

ent d

atase

t



in BE

V seg

menta

tion 

and d

etect

ion f

or dr

iving

, as 

the t

estin

g gro

und f

or MA

GICDR

IVE. 

We ad

here



to th

e off

icial

 conf

igura

tion,

 util

izing

 700 

stree

t-vie

w sce

nes f

or tr

ainin

g and

 150 

for v

alida

tion.



Our b

aseli

nes a

re BE

VGen 

(Swer

dlow 

et al

., 20

23) a

nd BE

VCont

rol (

Yang 

et al

., 20

23a),

 both



recen

t pro

posit

ions 

for s

treet

 view

 gene

ratio

n. Ou

r met

hod c

onsid

ers 1

0 obj

ect c

lasse

s and

 8 ro

ad



class

es, s

urpas

sing 

the b

aseli

ne mo

dels 

in di

versi

ty. A

ppend

ix B 

holds

 addi

tiona

l det

ails.



数据集和基

线。我们采

用 nuS

cenes

 数据集(

Caesa

r et 

al.,2

020)，

这是一个在

 BEV 

分割和



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



×



× ×



×



×



× ×



×



×



×



×



×



驾驶检测中

流行的数据

集，作为 

MAGIC

DRIVE

 的测试场

。我们坚持

官方配置，

利用 70

0 个街



景场景进行

训练，15

0 个场景

进行验证。

我们的基线

是 BEV

Gen(S

werdl

ow et

 al.,

2023)

和



BEVCo

ntrol

(Yang

 et a

l.,20

23a)，

都是街景生

成的近期命

题。我们的

方法考虑了

 10 个

对



象类和 8

 个道路类

，在多样性

上超过了基

线模型。附

录 B 保

存其他详细

信息。



Evalu

ation

 Metr

ics. 

We ev

aluat

e bot

h rea

lism 

and c

ontro

llabi

lity 

for s

treet

 view

 gene

ratio

n.



Real-

 ism 

is ma

inly 

measu

red u

sing 

Fre´c

het I

ncept

ion D

istan

ce (F

ID), 

refle

cting

 imag

e syn

thesi

s



quali

ty. F

or co

ntrol

labil

ity, 

MAGIC

DRIVE

 is e

valua

ted t

hroug

h two

 perc

eptio

n tas

ks: B

EV



segme

ntati

on an

d 3D 

objec

t det

ectio

n, wi

th CV

T (Zh

ou & 

Kra¨h

enbu¨

hl, 2

022) 

and B

EVFus

ion



(Liu 

et al

., 20

23a) 

as pe

rcept

ion m

odels

, res

pecti

vely.

 Both

 of t

hem a

re re

nowne

d for

 thei

r



perfo

rmanc

e in 

each 

task.

 Firs

tly, 

we ge

nerat

e ima

ges a

ligne

d wit

h the

 vali

datio

n set

 anno

tatio

ns



and u

se pe

rcept

ion m

odels

 pre-

train

ed wi

th re

al da

ta to

 asse

ss im

age q

ualit

y and

 cont

rol a

ccura

cy.



Then,

 data

 is g

enera

ted b

ased 

on th

e tra

ining

 set 

to ex

amine

 the 

suppo

rt fo

r tra

ining

 perc

eptio

n



model

s as 

data 

augme

ntati

on.



评估指标。

我们评估了

街景生成的

真实性和可

控性。真实

 ism 

主要使用 

Fre c

het 初

始距离



(FID)

来衡量，反

映图像合成

质量。对于

可控性，M

AGICD

RIVE 

通过两个感

知任务进行

评



估:BEV

 分割和 

3D 对象

检测，CV

T(Zho

u & K

ra¨he

nbu¨h

l,202

2)和 B

EVFus

ion(L

iu et



al.,2

023a)

分别作为感

知模型。他

们两人都因

在每项任务

中的表现而

闻名。首先

，我们生



成与验证集

注释对齐的

图像，并使

用用真实数

据预训练的

感知模型来

评估图像质

量和控制



准确性。然

后，基于训

练集生成数

据，以检查

对作为数据

扩充的训练

感知模型的

支持。



Model

 Setu

p. Ou

r MAG

ICDRI

VE ut

ilize

s pre

-trai

ned w

eight

s fro

m Sta

ble D

iffus

ion v

1.5, 

train

ing o

nly n

ewly 

added

 para

meter

s. Pe

r Zha

ng et

 al. 

(2023

a), a

 trai

nable

 UNet

 enco

der i

s cre

ated 

for



Emap.

 New 

param

eters

, exc

ept f

or th

e zer

o-ini

t mod

ule a

nd th

e cla

ss to

ken, 

are r

andom

ly



initi

alize

d. We

 adop

t two

 reso

lutio

ns to

 reco

ncile

 disc

repan

cies 

in pe

rcept

ion t

asks 

and b

aseli

nes:



224 4

00 (0

.25do

wn-sa

mple)

 foll

owing

 BEVG

en an

d for

 CVT 

model

 supp

ort, 

and a

 high

er



272 7

36 (0

.5 do

wn-sa

mple)

 for 

BEVFu

sion 

suppo

rt. U

nless

 stat

ed ot

herwi

se, i

mages

 are



sampl

ed us

ing t

he Un

iPC (

Zhao 

et al

., 20

23) s

chedu

ler f

or 20

 step

s wit

h CFG

 at 2

.0.



模型设置。

我们的 M

AGICD

RIVE 

利用 St

able 

Diffu

sion 

v1.5 

版的预训练

权重，仅训

练新添



加的参数。

每个 Zh

ang e

t al.

（2023

a)，为 

Emap 

创建一个可

训练的 U

Net 编

码器。除了



zero-

init 

模块和类标

记之外，新

的参数是随

机初始化的

。我们采用

两种解决方

案来协调



感知任务和

基线中的差

异:BEV

Gen 和

 CVT 

模型支持后

的 224

 400(

0.25 

下采样)，

以及



BEVFu

sion 

支持的更高

的 272

 736 

(0.5 

下采样)。

除非另有说

明，否则使

用 Uni

PC(Zh

ao et



al.,2

023)C

FG 为 

2.0 的

 20 步

调度程序。



10.3 

MAIN 

RESUL

TS



10.4 

主要结果



Reali

sm an

d Con

troll

abili

ty Va

lidat

ion. 

We as

sess 

MAGIC

DRIVE

’s ca

pabil

ity t

o cre

ate r

ealis

tic



stree

t-vie

w ima

ges w

ith t

he an

notat

ions 

from 

the n

uScen

es va

lidat

ion s

et. A

s sho

wn by

 Tabl

e 1,



MAGIC

DRIVE

 outp

erfor

ms ot

hers 

in im

age q

ualit

y, yi

eldin

g not

ably 

lower

 FID 

score

s. Re

gardi

ng



contr

ollab

ility

, ass

essed

 via 

BEV s

egmen

tatio

n tas

ks, M

AGICD

RIVE 

equal

s or 

excee

ds ba

selin

e res

ults 

at 22

4 400

 reso

lutio

n due

 to t

he di

stinc

t enc

oding

 desi

gn th

at en

hance

s veh

icle 

gener

ation



preci

sion.

 At 2

72 73

6 res

oluti

on, o

ur en

codin

g str

ategy

 adva

nceme

nts e

nhanc

e veh

icle 

mIoU 

perfo

rmanc

e. Cr

oppin

g lar

ge ar

eas n

egati

vely 

impac

ts ro

ad mI

oU on

 CVT.

 Howe

ver, 

our b

oundi

ng



box e

ncodi

ng ef

ficac

y is 

backe

d by 

BEVFu

sion’

s res

ults 

in 3D

 obje

ct de

tecti

on.



真实性和可

控性验证。

我们评估 

MAGIC

DRIVE

 使用 n

uScen

es 验证

集的注释创

建真实街景

图



像的能力。

如表所示 

1，MAG

ICDRI

VE 在图

像质量方面

优于其他产

品，FID

 分数明显

较低。关



于可控性，

通过 BE

V 分段任

务评估，M

AGICD

RIVE 

在 224

 400 

分辨率下等

于或超过基

线结



果，这是由

于独特的编

码设计提高

了车辆生成

精度。在 

272 7

36 分辨

率下，我们

的编码策



略进步增强

了车辆的 

mIoU 

性能。大面

积种植会对

 CVT 

上的道路 

mIoU 

产生负面影

响。然



而，BEV

Fusio

n 在 3

D 对象检

测中的结果

支持了我们

的边界框编

码效率。



BEV s

egmen

tatio

n 3D 

objec

t det

ectio

n



Metho

d Syn

thesi

s



↑



BEV s

egmen

tatio

n 3D 

objec

t det

ectio

n



Metho

d Syn

thesi

s



↑



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



Table

 1: C

ompar

ison 

of ge

nerat

ion f

ideli

ty wi

th dr

iving

-view

 gene

ratio

n met

hods.

 Cond

ition

s for



data 

synth

esis 

are f

rom n

uScen

es va

lidat

ion s

et. F

or ea

ch ta

sk, w

e tes

t the

 corr

espon

ding 

model

s



train

ed on

 the 

nuSce

nes t

raini

ng se

t. MA

GICDR

IVE s

urpas

ses a

ll ba

selin

es th

rough

out t

he ev

aluat

ion. 

/ ind

icate

s tha

t a h

igher

/lowe

r val

ue is

 bett

er. T

he be

st re

sults

 are 

in bo

ld, w

hile 

the s

econd



best 

resul

ts ar

e in 

under

lined

 ital

ic (w

hen o

ther 

metho

ds ar

e ava

ilabl

e).



表 1:生

成逼真度与

驱动视图生

成方法的比

较。数据合

成的条件来

自 nuS

cenes

 验证集。

对于



每个任务，

我们测试在

 nuSc

enes 

训练集上训

练的相应模

型。MAG

ICDRI

VE 在整

个评估过程

中



超越了所有

基线。/表

示值越高/

越低越好。

最好的结果

用粗体显示

，第二好的

结果用下划



线斜体显示

(当其他方

法可用时)

。



resol

ution

 FID↓

 Road

 mIoU

 ↑ Ve

hicle

 mIoU

 ↑ mA

P ↑ N

DS ↑



解决 FI

D↓ 米欧

路 车辆 

mIoU 

图↑ ND

S ↑



Oracl

e - -

 72.2

1 33.

66 35

.54 4

1.21



Oracl

e 224

×400 

- 72.

19 33

.61 2

3.54 

31.08



神谕 - 

- 72.

2



1



33.66

 35.5



4



41.2



1



神谕 22

4×400

 - 72

.1



9



33.61

 23.5



4



31.0



8



BEVGe

n 224

×



-



400 2

5.54 

50.20

 5.89

 - -



BEVCo

ntrol

 24.8

5 60.

80 26

.80 -

 -



贝夫根 2

24×-



400



25.5



4



50.2



0



5.89 

- -



饮料控制 

24.8



5



60.8



0



26.80

 - -



MAGIC

DRIVE



MAGIC

DRIVE



224×4

00



272×7

36



16.20



16.59



61.05



54.24



27.01



31.05



12.30



20.85



23.32



30.26



魔法驱动



魔法驱动



224×4

00



272×7

36



×



16.20



16.59



61.05



54.24



27.0



1



31.0



5



12.30



20.85



23.32



30.26



Incon

siste

nt



Wrong

 plac

e



不一致错误



的地方



Groun

d Tru

th



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



地面实况;

 真值



（机器学习

）



BEVCo

ntrol



饮料控制



Figur

e 5: 

Quali

tativ

e com

paris

on wi

th BE

VCont

rol o

n dri

ving 

scene

 from

 nuSc

enes 

valid

ation

 set.



We hi

ghlig

ht so

me ar

eas w

ith r

ectan

gles 

to ea

se co

mpari

son. 

Compa

red w

ith B

EVCon

trol,



gener

a- ti

ons f

rom M

AGICD

RIVE 

appea

r mor

e con

siste

nt in

 both

 back

groun

d and

 fore

groun

d.



图 nuS

cenes

 验证集中

驾驶场景与

 BEVC

ontro

l 的定性

比较。为了

便于比较，

我们用矩形

突



出显示了一

些区域。与

 BEVC

ontro

l 相比，

MAGIC

DRIVE

 的生成在

背景和前景

上都显得更

加一



致。



Train

ing S

uppor

t for

 BEV 

Segme

ntati

on an

d 3D 

Objec

t Det

ectio

n. MA

GICDR

IVE c

an pr

oduce



augme

nted 

data 

with 

accur

ate a

nnota

tion 

contr

ols, 

enhan

cing 

the t

raini

ng fo

r per

cepti

on ta

sks. 

For



BEV s

egmen

tatio

n, we

 augm

ent a

n equ

al nu

mber 

of im

ages 

as in

 the 

origi

nal d

atase

t, en

surin

g



con- 

siste

nt tr

ainin

g ite

ratio

ns an

d bat

ch si

zes f

or fa

ir co

mpari

sons 

to th

e bas

eline

. As 

shown

 in



Table

 3, M

AGICD

RIVE 

signi

fican

tly e

nhanc

es CV

T in 

both 

setti

ngs, 

outpe

rform

ing B

EVGen

,



which

 only

 marg

inall

y imp

roves

 vehi

cle s

egmen

tatio

n. Fo

r 3D 

objec

t det

ectio

n, we

 trai

n



BEVFu

sion 

model

s wit

h MAG

ICDRI

VE ’s

 synt

hetic

 data

 as a

ugmen

tatio

n. To

 opti

mize 

data



augme

ntati

on, w

e ran

domly

 excl

ude 5

0% of

 boun

ding 

boxes

 in e

ach g

enera

ted s

cene.

 Tabl

e 2



shows

 the 

advan

tageo

us im

pact 

of MA

GICDR

IVE ’

s dat

a in 

both 

CAM-o

nly (

C) an

d CAM

+LiDA

R



(C+L)

 sett

ings.

 It’s

 cruc

ial t

o not

e tha

t in 

CAM+L

iDAR 

setti

ngs, 

BEVFu

sion 

utili

zes b

oth



modal

ities

 for 

objec

t det

ectio

n, re

quiri

ng mo

re pr

ecise

 imag

e gen

erati

on du

e to 

LiDAR

 data



incor

porat

ion. 

Never

thele

ss, M

AGICD

RIVE’

s syn

theti

c dat

a int

egrat

es se

amles

sly w

ith L

iDAR



input

s, hi

ghlig

hting

 the 

data’

s hig

h fid

elity

.



对 BEV

 分割和 

3D 对象

检测的训练

支持。MA

GICDR

IVE 可

以通过精确

的注释控制

产生增强数



据，从而增

强感知任务

的训练。对

于 BEV

 分割，我

们增加了与

原始数据集

中相同数量

的图



像，以确保

持续的训练

迭代和批量

大小，以便

与基线进行

公平的比较

。如表中所

示



3，MAG

ICDRI

VE 在这

两种设置下

显著增强了

 CVT，

优于仅略微

改善车辆分

割的 BE

VGen。

对于



3D 对象

检测，我们

用 MAG

ICDRI

VE 的合

成数据作为

增强来训练

 BEVF

usion

 模型。为

了优化数



据扩充，我

们在每个生

成的场景中

随机排除 

50%的边

界框。桌子

 2 显示

了 MAG

ICDRI

VE 数据

在



仅 CAM

(C)和 

CAM+L

iDAR 

(C+L)

设置中的有

利影响。值

得注意的是

，在 CA

M+激光雷

达设置



中，BEV

Fusio

n 利用两

种模式进行

物体检测，

由于激光雷

达数据的合

并，需要更

精确的图像

生



成。然而，

MAGIC

DRIVE

 的合成数

据与激光雷

达输入无缝

集成，突出

了数据的高

保真度。



10.5 

QUALI

TATIV

E EVA

LUATI

ON



10.6 

定性评价



Compa

rison

 with

 Base

lines

. We 

asses

sed M

AGICD

RIVE 

again

st tw

o bas

eline

s, BE

VGen 

and



BEVCo

ntrol

, syn

thesi

zing 

multi

-came

ra vi

ews f

or th

e sam

e val

idati

on sc

enes 

(the 

compa

rison



with 

BEVGe

n is 

in th

e App

endix

 D). 

Figur

e 5 i

llust

rates

 that

 MAGI

CDRIV

E gen

erate

s ima

ges



marke

dly s

uperi

or in

 qual

ity t

o BEV

Contr

ol, p

artic

ularl

y exc

ellin

g in 

accur

ate o

bject

 posi

tioni

ng



and m

ain- 

taini

ng co

nsist

ency 

in st

reet 

views

 for 

backg

round

s and

 obje

cts. 

Such 

perfo

rmanc

e



prima

rily 

stems

 from

 MAGI

CDRIV

E ’s 

bound

ing b

ox en

coder

 and 

its c

ross-

view 

atten

tion 

modul

e.



与基线的比

较。我们根

据两条基线

(BEVG

en 和 

BEVCo

ntrol

)对 MA

GICDR

IVE 进

行了评估，

合



成了相同验

证场景的多

摄像机视图

(与 BE

VGen 

的比较见附

录 D).

数字 5 

说明 MA

GICDR

IVE 生



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



1



成的图像质

量明显优于

 BEVC

ontro

l，特别是

在精确的对

象定位和保

持背景和对

象的街景一



致性方面表

现出色。这

样的性能主

要源于 M

AGICD

RIVE 

的边界框编

码器和它的

跨视图注意

模



块。



Multi

-leve

l Con

trols

. The

 desi

gn of

 MAGI

CDRIV

E int

roduc

es mu

lti-l

evel 

contr

ols t

o str

eet-v

iew



gener

ation

 thro

ugh s

epara

tion 

encod

ing. 

This 

secti

on de

monst

rates

 the 

capab

iliti

es of

 MAGI

CDRIV

E by 

explo

ring 

three

 cont

rol s

ignal

 leve

ls: s

cene 

level

 (tim

e of 

day a

nd we

ather

), ba

ckgro

und



level

 (BEV

 map 

alter

ation

s and

 cond

ition

al vi

ews),

 and 

foreg

round

 leve

l (ob

ject 

orien

tatio

n and



dele-

 tion

). As

 illu

strat

ed in

 Figu

re 1,

 Figu

re 6,

 and 

Appen

dix E

, MAG

ICDRI

VE ad

eptly



accom

modat

es al

terat

ions 

at ea

ch le

vel, 

maint

ainin

g mul

ti-ca

mera 

consi

stenc

y and

 high

 real

ism i

n



gener

ation

.



多级控制。

MAGIC

DRIVE

 的设计通

过分离编码

将多级控制

引入街景生

成。本节通

过探索三个



控制信号级

别来展示 

MAGIC

- DRI

VE 的功

能:场景级

别(一天中

的时间和天

气)、背景

级别



(BEV 

地图变更和

条件视图)

和前景级别

(对象定向

和删除)。

如图所示 

1，图 6

 和附录



E，MAG

ICDRI

VE 巧妙

地适应了每

个级别的变

化，保持了

多相机的一

致性和高度

的真实感。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



Data 

Vehic

le Ro

ad



mIoU 

↑mIoU

 ↑



Data 

Vehic

le Ro

ad



mIoU 

↑mIoU

 ↑



Table

 2: C

ompar

ison 

about

 supp

ort f

or 3D

 obje

ct



detec

tion 

model

 (i.e

., BE

VFusi

on). 

MAGIC

DRIVE



gener

ates 

272×7

36 im

ages 

for a

ugmen

tatio

n. Re

sults

 are 

repor

ted o

n the

 nuSc

enes 

valid

ation

 set.



表 2:关

于支持 3

D 对象检

测模型(即

 BEVF

usion

)



的比较。M

AGICD

RIVE 

生成 27

2×736

 的图像用

于



增强。在 

nuSce

nes 验

证集上报告

结果。



Table

 3: C

ompar

ison 

about

 supp

ort f

or



BEV s

egmen

tatio

n mod

el (i

.e., 

CVT).



Resul

ts ar

e rep

orted

 by t

estin

g on 

the



nuSce

nes v

alida

tion 

set.



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



Gener

ation

 from



Backg

round

 leve

l: co

nditi

on on

 one 

given

 view

 (ann

otate

d wit

h red

 box)



Gener

ation

 from



Foreg

round

 leve

l: ro

tate 

each 

vehic

le 18

0 deg

ree (

head 

to ta

il) (

we hi

ghlig

ht so

me ro

tated

 vehi

cles 

for c

ompar

ison)



Foreg

round

 leve

l: de

lete 

50% o

bject

s (we

 show

 the 

DELET

ED bo

xes a

nd hi

ghlig

ht so

me ar

eas f

or co

mpari

son)



Gener

ation

 from



Backg

round

 leve

l: co

nditi

on on

 one 

given

 view

 (ann

otate

d wit

h red

 box)



Gener

ation

 from



Foreg

round

 leve

l: ro

tate 

each 

vehic

le 18

0 deg

ree (

head 

to ta

il) (

we hi

ghlig

ht so

me ro

tated

 vehi

cles 

for c

ompar

ison)



Foreg

round

 leve

l: de

lete 

50% o

bject

s (we

 show

 the 

DELET

ED bo

xes a

nd hi

ghlig

ht so

me ar

eas f

or co

mpari

son)



表 3:对

 BEV 

分段模型(

即 CVT

)支持的比

较。通



过在 nu

Scene

s 验证集

上进行测试

来报告结



果。



Modal

ity D

ata m

AP ↑ 

NDS ↑



形式 数据

 图↑ N

DS ↑



Figur

e 6: 

Showc

ase f

or mu

lti-l

evel 

contr

ol wi

th MA

GICDR

IVE. 

We sh

ow ba

ckgro

und-l

evel 

and



foreg

round

-leve

l con

trol 

separ

ately

 with

 diff

erent

 cond

ition

s. Al

l sce

nes a

re ba

sed o

n the



nuSce

nes v

alida

tion 

set. 

More 

resul

ts ar

e in 

Appen

dix E

.



图 6:使

用 MAG

ICDRI

VE 进行

多级控制的

展示。我们

分别展示了

不同条件下

的背景级和

前台级



控件。所有

场景都基于

 nuSc

enes 

验证集。更

多结果见附

录 E。



10.7 

EXTEN

SION 

TO VI

DEO G

ENERA

TION



10.8 

视频生成的

扩展



We de

monst

rate 

the e

xtens

ibili

ty of

 MAGI

CDRIV

E to 

video

 gene

ratio

n by 

fine-

tunin

g it 

on



nuSce

nes v

ideos

. Thi

s inv

olves

 modi

fying

 self

-atte

ntion

 to S

T-Att

n (Wu

 et a

l., 2

023a)

, add

ing a



tempo

ral a

ttent

ion m

odule

 to e

ach t

ransf

ormer

 bloc

k (Fi

gure 

7 lef

t), a

nd tu

ning 

the m

odel 

on 7-



frame

 clip

s wit

h onl

y the

 firs

t and

 the 

last 

frame

s hav

ing b

oundi

ng bo

xes. 

We sa

mple 

initi

al no

ise



indep

enden

tly f

or ea

ch fr

ame u

sing 

the U

niPC 

(Zhao

 et a

l., 2

023) 

sampl

er fo

r 20 

steps

 and



illus

trate

 an e

xampl

e in 

Figur

e 7 r

ight.



我们通过在

 nuSc

enes 

视频上进行

微调，展示

了 MAG

ICDRI

VE 对视

频生成的可

扩展性。这

涉



及到修改自

我注意到圣

 Attn

(Wu e

t al.

,2023

a)，向每

个变换器块

添加时间注

意模块(图

 7



左)，并在

只有第一帧

和最后一帧

具有边界框

的 7 帧

剪辑上调整

模型。我们

使用 Un

iPC(Z

hao



et al

.,202

3)采样器

为 20 

步，并在图

中举例说明

 7 没错

。



Furth

ermor

e, by

 util

izing

 the 

inter

polat

ed an

notat

ions 

from 

ASAP 

(Wang

 et a

l., 2

023c)

 like

 Driv

eDrea

mer (

Wang 

et al

., 20

23b),

 Magi

cDriv

e can

 be e

xtend

ed to

 16-f

rame 

video

 gene

ratio

n at 

12Hz



train

ed on

 Nvid

ia V1

00 GP

Us. M

ore r

esult

s (e.

g., v

ideo 

visua

lizat

ion) 

can b

e fou

nd on

 our



websi

te.



此外，通过

利用来自 

ASAP 

的内插注释

(Wang

 et a

l.,20

23c)就

像开车的梦

想家(Wa

ng et



al.,2

023b)

，Magi

cDriv

e 可以扩

展到在 N

vidia

 V100

 GPUs

 上训练的

 12Hz

 的 16

 帧视频生



w/o s

ynthe

tic



data



32.88

 37.8

1



C



w/ MA

GICDR

IVE 3

5.40 

+2.52

 39.7

6 +1.

95



w/o s

ynthe

tic d

ata 3

6.00 

74.30



w/o s

ynthe

tic



data



32.88

 37.8

1



C



w/ MA

GICDR

IVE 3

5.40 

+2.52

 39.7

6 +1.

95



w/o s

ynthe

tic d

ata 3

6.00 

74.30



w/o s

ynthe

tic



data



65.40

 69.5

9



C+L



w/ MA

GICDR

IVE 6

7.86 

+2.46

 70.7

2 +1.

13



w/ BE

VGen 

36.60

 +0.6

0 71.

90 -2

.40



w/ MA

GICDR

IVE 4

0.34 

+4.34

 79.5

6 +5.

26



w/o s

ynthe

tic



data



65.40

 69.5

9



C+L



w/ MA

GICDR

IVE 6

7.86 

+2.46

 70.7

2 +1.

13



w/ BE

VGen 

36.60

 +0.6

0 71.

90 -2

.40



w/ MA

GICDR

IVE 4

0.34 

+4.34

 79.5

6 +5.

26



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



成。更多结

果(如视频

可视化)可

在我们的网

站上找到。



11 AB

LATIO

N STU

DY



12 消融

研究



Bound

ing B

ox En

codin

g. MA

GICDR

IVE u

tiliz

es se

parat

e enc

oders

 for 

bound

ing b

oxes 

and r

oad



maps.

 To d

emons

trate

 the 

effic

acy, 

we tr

ain a

 Cont

rolNe

t (Zh

ang e

t al.

, 202

3a) t

hat t

akes 

the B

EV



map w

ith b

oth r

oad a

nd ob

ject 

seman

tics 

as a 

condi

tion 

(like

 BEVG

en), 

denot

ed as

 “w/o

 Ebox

” in



Table

 4. O

bject

s in 

BEV m

aps a

re re

lativ

ely s

mall,

 whic

h req

uire 

separ

ate E

box f

or ac

curat

e



vehic

le an

notat

ions,

 as s

hown 

by th

e veh

icle 

mIoU 

perfo

rmanc

e gap

. App

lying

 visi

ble o

bject



filte

r fvi

z sig

nific

antly

 impr

oves 

both 

road 

and v

ehicl

e mIo

U by 

reduc

ing t

he op

timiz

ation

 burd

en.



A MAG

- ICD

RIVE 

varia

nt in

corpo

ratin

g Ebo

x wit

h BEV

 of r

oad a

nd ob

ject 

seman

tics 

didn’

t



enhan

ce pe

rfor-

 manc

e, em

phasi

zing 

the i

mport

ance 

of in

tegra

ting 

diver

se in

forma

tion 

throu

gh



diffe

rent 

strat

egies

.



边界框编码

。MAGI

CDRIV

E 对边界

框和路线图

使用单独的

编码器。为

了证明其有

效性，我们



训练了一个

控制网络(

Zhang

 et a

l.,20

23a)将

带有道路和

对象语义的

 BEV 

映射作为条

件(如



BEVGe

n)，在表

中表示为“

w/o E

box”4

。BEV 

贴图中的对

象相对较小

，这需要单

独的 Eb

ox



来进行精确

的车辆注释

，如车辆 

mIoU 

性能差距所

示。应用可

见对象过滤

器 fvi

z 通过减

少



优化负担显

著改善了道

路和车辆的

 mIoU

。将 Eb

ox 与 

BEV o

f roa

d 和 o

bject

 语义相结

合的



MAG- 

ICDRI

VE 变体

并没有提高

性能，而是

强调了通过

不同策略集

成不同信息

的重要性。



Effec

t of 

Class

ifier

-free

 Guid

ance.

 We f

ocus 

on th

e two

 most

 cruc

ial c

ondit

ions,

 i.e.

 obje

ct



boxes

 and 

road 

maps,

 and 

analy

ze ho

w CFG

 affe

cts t

he pe

rform

ance 

of ge

nerat

ion. 

We ch

ange



CFG f

rom



无分类器制

导的效果。

我们重点讨

论了两个最

关键的条件

，即对象盒

和路线图，

并分析了



CFG 如

何影响生成

的性能。我

们将 CF

G 从



1.5 t

o 4.0

 and 

plot 

the c

hange

 of v

alida

tion 

resul

ts fr

om CV

T in 

Figur

e 8. 

First

ly, b

y inc

reasi

ng



1.5 至

 4.0，

并将 CV

T 验证结

果的变化绘

制在图中 

8。首先，

通过增加



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



Key +

 Prev

. fra

mes S

T-Att

n QKV



<CAM>

 <TXT

> <BO

Xes> 

cross

-attn



Left 

Neigh

bor



cross

-view

 attn



Right

 Neig

hbor



Train

able 

Froze

n tem

p att

n



Key +

 Prev

. fra

mes



ST-At

tn QK

V



<CAM>

 <TXT

> <BO

Xes> 

cross

-attn



Left 

Neigh

bor



cross

-view

 attn



Right

 Neig

hbor



Train

able 

Froze

n



temp 

attn



Tansf

ormer

 Bloc

k



变压器组



Keyfr

ame



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



{



{



键架



Keyfr

ame



键架



Figur

e 7: 

Exten

d MAG

ICDRI

VE to

 vide

o gen

erati

on. L

eft: 

chang

es in

 tran

sform

er bl

ock f

or vi

deo



gener

ation

. Rig

ht: r

esult

 of v

ideo 

gener

ation

. Onl

y key

frame

s hav

e bou

nding

 box 

contr

ol.



图 7:将

 MAGI

CDRIV

E 扩展到

视频生成。

左图:用于

视频生成的

变压器模块

的变化。右

图:视频生

成



的结果。只

有关键帧具

有边界框控

制。



Table

 4: A

blati

on of

 the 

separ

ate b

ox en

coder

. Eva

luati

on



resul

ts ar

e fro

m CVT

 on t

he sy

nthet

ic nu

Scene

s



valid

ation

 set,

 with

out M

 = 0 

in CF

G sca

le = 

2.



MAGIC

DRIVE

 has 

bette

r con

troll

abili

ty an

d kee

ps im

age



quali

ty.



表 4:独

立盒式编码

器的烧蚀。

评估结果来

自合成



nuSce

nes 验

证集上的 

CVT，在

 CFG 

标度= 2

 中没有 

M =



0。MAG

ICDRI

VE 可控

性更好，保

持画质。



Metho

d



w/o E

box



方法无



Ebox



w/o f

viz



不含 fv

iz



w/ Eb

ox & 

mapob

j



Ours



带有 Eb

ox 和



mapob

j Our

s



Road



mIoU



↑



欧



路



Vehic

le mI

oU ↑



FI



FID



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



{



{



Groun

dTrut

h Gro

undTr

uth



+ nig

ht + 

snow



(a) n

ot da

rk en

ough 

(b) u

nseen

 weat

her



Groun

dTrut

h Gro

undTr

uth



+ nig

ht + 

snow



(a) n

ot da

rk en

ough 

(b) u

nseen

 weat

her



车辆 mI

oU



Figur

e 8: 

Effec

t of 

CFG o

n dif

feren

t



condi

tions

 to e

ach m

etric

s.



图 8:不

同条件下 

CFG 对

每个指标的

影



响。



CFG s

cale,

 FID 

degra

des d

ue to

 nota

ble c

hange

s in 

contr

ast a

nd sh

arpne

ss, a

s see

n in 

previ

ous



stud-

 ies 

(Chen

 et a

l., 2

023c)

. Sec

ondly

, ret

ainin

g the

 same

 map 

for b

oth c

ondit

ional

 and



uncon

ditio

nal i

nfere

nce e

limin

ates 

CFG’s

 effe

ct on

 the 

map c

ondit

ion. 

As sh

own b

y blu

e lin

es of



Figur

e 8, 

incre

as- i

ng CF

G sca

le re

sults

 in t

he hi

ghest

 vehi

cle m

IoU a

t CFG

=2.5,

 but 

the r

oad



mIoU 

keeps

 decr

easin

g. Th

irdly

, wit

h M =

0for 

uncon

ditio

nal i

nfere

nce i

n CFG

, roa

d mIo

U



signi

fican

tly i

ncrea

ses.



CFG 标

度，FID

 由于对比

度和锐度的

显著变化而

降低，如以

前的研究中

所见(Ch

en et



al.,2

023c)

.其次，为

条件和无条

件推理保留

相同的映射

消除了 C

FG 对映

射条件的影

响。如



图中的蓝线

所示 8，

增加 CF

G 比例导

致 CFG

=2.5 

时的最高车

辆 mIo

U，但道路

 mIoU

 持续下



降。第三，

对于 CF

G 中的无

条件推断，

M =，r

oad m

IoU 显

著增加。



Howev

er, i

t sli

ghtly

 degr

ades 

the g

uidan

ce on

 vehi

cle g

enera

tion.

 As m

entio

ned i

n Sec

tion 

4.3,



然而，它稍

微降低了对

车辆生成的

指导。如第

节所述 4

.3,



CFG c

omple

xity 

incre

ases 

with 

more 

condi

tions

. Des

pite 

simpl

ifyin

g tra

ining

, var

ious 

CFG



choic

es ex

ist d

uring

 infe

rence

. We 

leave

 the 

in-de

pth i

nvest

igati

on fo

r thi

s cas

e as 

futur

e wor

k.



CFG 的

复杂性随着

条件的增加

而增加。尽

管简化了训

练，但在推

断过程中存

在各种 C

FG 选



择。我们把

对这一案件

的深入调查

作为今后的

工作。



13 CO

NCLUS

ION



14 结论



This 

paper

 pres

ents 

MAGIC

DRIVE

, a n

ovel 

frame

work 

to en

code 

multi

ple g

eomet

ric c

ontro

ls fo

r



high-

quali

ty mu

lti-c

amera

 stre

et vi

ew ge

nerat

ion. 

With 

the s

epara

tion 

encod

ing d

esign

, MAG

ICDRI

VE fu

lly u

tiliz

es ge

ometr

ic in

forma

tion 

from 

3D an

notat

ions 

and r

ealiz

es ac

curat

e sem

antic



contr

ol fo

r str

eet v

iews.

 Besi

des, 

the p

ropos

ed cr

oss-v

iew a

ttent

ion m

odule

 is s

imple

 yet 

effec

tive



in gu

arant

eeing

 cons

isten

cy ac

ross 

multi

-came

ra vi

ews. 

As ev

idenc

ed by

 expe

rimen

ts, t

he



gener

ation

s fro

m MAG

ICDRI

VE sh

ow hi

gh re

alism

 and 

fidel

ity t

o 3D 

annot

ation

s. Mu

ltipl

e



contr

ols e

quipp

ed MA

GICDR

IVE w

ith i

mprov

ed ge

neral

izabi

lity 

for t

he ge

nerat

ion o

f nov

el st

reet



views

. Mea

nwhil

e, MA

GICDR

IVE c

an be

 used

 for 

data 

augme

ntati

on, f

acili

tatin

g the

 trai

ning 

for



perce

ption

 mode

ls on

 both

 BEV 

segme

ntati

on an

d 3D 

objec

t det

ectio

n tas

ks.



本文介绍了

 MAGI

CDRIV

E，一个新

的框架来编

码高质量的

多摄像机街

景生成的多

种几何控



制。MAG

IC- D

RIVE 

采用分离编

码设计，充

分利用 3

D 标注的

几何信息，

实现对街景

的精确语



义控制。此

外，所提出

的跨视角注

意模块简单

而有效地保

证了跨多摄

像机视角的

一致性。



实验证明，

MAGIC

DRIVE

 的各代产

品对 3D

 注释表现

出高度的真

实性和保真

度。多个控

件为



MAGIC

DRIVE

 配备了改

进的泛化能

力，可生成

新颖的街道

视图。同时

，MAGI

CDRIV

E 可用于

数



据增强，有

助于在 B

EV 分割

和 3D 

对象检测任

务中训练感

知模型。



Limit

ation

 and 

Futur

e Wor

k. We

 show

 fail

ure



cases

 from

 MAGI

CDRIV

E in 

Figur

e 9. 

Altho

ugh



MAGIC

DRIVE

 can 

gener

ate n

ight 

views

, the

y are



not a

s dar

k as 

real 

image

s (as

 in F

igure

 9a).

 This



may b

e due

 to t

hat d

iffus

ion m

odels

 are 

hard 

to



gener

ate t

oo da

rk im

ages 

(Gutt

enber

g, 20

23). 

Figur

e 9b 

shows

 that

 MAGI

CDRIV

E can

not g

enera

te



局 限 性

 和 未 

来 工 作

 。 我 

们 在 图

 中 显 

示 了



MAGIC

DRIVE

 的失败案

例 9。虽

然 MAG

ICDRI

VE



可以生成夜

景，但它们

不像真实图

像那样暗



(如图 9

a)。这可

能是由于扩

散模型难以

生成



太 暗 的

 图 像 

(Gutt

enber

g,202

3). 图

 9b 表

 示



MAGIC

DRIVE

 无法生成



18.06

 58.3

1 5.5

0



14.67

 56.4

6 24.

73



14.70

 56.0

4 26.

20



14.46

 59.3

1 27.

13



18.06

 58.3

1 5.5

0



14.67

 56.4

6 24.

73



14.70

 56.0

4 26.

20



14.46

 59.3

1 27.

13



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



unsee

n wea

thers

 for 

nuSce

nes. 

Futur

e wor

k may



未知的天气

。未来的工

作可以



Figur

e 9: 

Failu

re ca

ses o

f MAG

ICDRI

VE.



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



图 MAG

ICDRI

VE 的失

败案例。



focus

 on h

ow to

 impr

ove t

he cr

oss-d

omain

 gene

raliz

ation

 abil

ity o

f str

eet v

iew g

enera

tion.



重点研究如

何提高街景

生成的跨域

概括能力。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



Ackno

wledg

ement

. Thi

s wor

k is 

suppo

rted 

in pa

rt by

 the 

Gener

al Re

searc

h Fun

d (GR

F) of

 Hong



Kong 

Resea

rch G

rants

 Coun

cil (

RGC) 

under

 Gran

t No.

 1420

3521,

 in p

art b

y the

 CUHK

 SSFC

RS



fundi

ng No

. 313

6023,

 and 

in pa

rt by

 the 

Resea

rch M

atchi

ng Gr

ant S

cheme

 unde

r Gra

nt No

.



71069

37, 8

60113

0, an

d 860

1440.

 We g

ratef

ully 

ackno

wledg

e the

 supp

ort o

f Min

dSpor

e, CA

NN



(Comp

ute A

rchit

ectur

e for

 Neur

al Ne

twork

s) an

d Asc

end A

I Pro

cesso

r use

d for

 this

 rese

arch.

 This



resea

rch h

as be

en ma

de po

ssibl

e by 

fundi

ng su

pport

 from

 the 

Resea

rch G

rants

 Coun

cil o

f Hon

g



Kong 

throu

gh th

e Res

earch

 Impa

ct Fu

nd pr

oject

 R600

3-21.



致谢。本研

究得到了 

RGC 研

究资助局一

般研究基金

(GRF)

项目(14

20352

1)、CU

HK 科学

研究资



助计划项目

(3136

023)和

研究配对补

助金项目(

71069

37、86

01130

 和 86

01440

)的资助。

我们



非常感谢 

MindS

pore、

CANN(

神经网络计

算架构)和

 Asce

nd AI

 处理器对

本次研究的

支持。这



项研究得到

了香港研究

资助局通过

研究影响力

基金项目 

R6003

-21 的

资助。



REFER

ENCES



参考



Tim B

rooks

, Ale

ksand

er Ho

lynsk

i, an

d Ale

xei A

 Efro

s. In

struc

tpix2

pix: 

Learn

ing t

o fol

low i

mage



editi

ng in

struc

tions

. In 

CVPR,

 2023

.



蒂姆·布鲁

克斯、亚历

山大·霍林

斯基和阿列

克谢·埃夫

罗斯。In

struc

tpix2

pix:学

习遵



循图像编辑

说明。在 

CVPR，

2023 

年。



Holge

r Cae

sar, 

Varun

 Bank

iti, 

Alex 

H Lan

g, So

urabh

 Vora

, Ven

ice E

rin L

iong,

 Qian

g Xu,

 Anus

h



Krish

nan, 

Yu Pa

n, Gi

ancar

lo Ba

ldan,

 and 

Oscar

 Beij

bom. 

nusce

nes: 

A mul

timod

al da

taset

 for



auton

omous

 driv

ing. 

In CV

PR, 2

020.



霍尔格·凯

撒、瓦伦·

班基蒂、亚

历克斯·H

·朗、苏拉

布·沃拉、

威尼斯·艾

琳·莱昂、



徐强、阿努

什·克里希

南、潘宇、

贾恩卡洛·

巴尔丹和奥

斯卡·贝伊

邦。nus

cenes

:用于



自动驾驶的

多模态数据

集。在 2

020 年

的 CVP

R。



Kai C

hen, 

Lanqi

ng Ho

ng, H

ang X

u, Zh

enguo

 Li, 

and D

it-Ya

n Yeu

ng. M

ultis

iam: 

Self-

super

vised



multi

-inst

ance 

siame

se re

prese

ntati

on le

arnin

g for

 auto

nomou

s dri

ving.

 In I

CCV, 

2021.



程凯、洪蓝

青、徐航、

李振国和杨

迪燕。Mu

ltisi

am:用于

自动驾驶的

自监督多示

例暹罗表



示学习。2

021 年

在 ICC

V。



Kai C

hen, 

Zhili

 Liu,

 Lanq

ing H

ong, 

Hang 

Xu, Z

hengu

o Li,

 and 

Dit-Y

an Ye

ung. 

Mixed



autoe

ncode

r for

 self

-supe

rvise

d vis

ual r

epres

entat

ion l

earni

ng. I

n CVP

R, 20

23a.



、刘志立、

洪蓝青、、

、杨迪燕。

用于自监督

视觉表征学

习的混合自

动编码器。

2023 

年在



CVPR。



Kai C

hen, 

Chunw

ei Wa

ng, K

uo Ya

ng, J

ianhu

a Han

, Lan

qing 

Hong,

 Fei 

Mi, H

ang X

u, Zh

engyi

ng



Liu, 

Wenyo

ng Hu

ang, 

Zheng

uo Li

, Dit

-Yan 

Yeung

, Lif

eng S

hang,

 Xin 

Jiang

, and

 Qun 

Liu.



Gain-

 ing 

wisdo

m fro

m set

backs

: Ali

gning

 larg

e lan

guage

 mode

ls vi

a mis

take 

analy

sis. 

arXiv



prepr

int a

rXiv:

2310.

10477

, 202

3b.



、王纯薇、

杨阔、韩建

华、洪蓝青

、费米、徐

航、刘正英

、黄文永、

李振国、杨

迪燕、尚



立峰、新疆

和刘群。从

挫折中获得

智慧:通过

错误分析校

准大型语言

模型。ar

Xiv 预

印本



arXiv

:2310

.1047

7，202

3b。



Kai C

hen, 

Enze 

Xie, 

Zhe C

hen, 

Lanqi

ng Ho

ng, Z

hengu

o Li,

 and 

Dit-Y

an Ye

ung. 

Integ

ratin

g geo

metri

c con

trol 

into 

text-

to-im

age d

iffus

ion m

odels

 for 

high-

quali

ty de

tecti

on da

ta ge

nerat

ion v

ia



text 

promp

t. ar

Xiv p

repri

nt ar

Xiv:2

306.0

4607,

 2023

c.



、谢恩泽、

、洪蓝青、

、杨迪燕。

将几何控制

集成到文本

到图像扩散

模型中，通

过文本提



示生成高质

量的检测数

据。arX

iv 预印

本 arX

iv:23

06.04

607，2

023c。



Yaran

 Chen

, Hao

ran L

i, Ru

iyuan

 Gao,

 and 

Dongb

in Zh

ao. B

oost 

3-d o

bject

 dete

ction

 via 

point



cloud

s seg

menta

tion 

and f

used 

3-d g

iou-l

1 los

s. IE

EE TN

NLS, 

2020.



陈雅然，，

高瑞元，赵

。通过点云

分割和融合

的三维 g

iou-l

1 损失增

强三维物体

检



测。IEE

E TNN

LS，20

20。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



2



Patri

ck Es

ser, 

Robin

 Romb

ach, 

and B

jorn 

Ommer

. Tam

ing t

ransf

ormer

s for

 high

-reso

lutio

n ima

ge



synth

esis.

 In C

VPR, 

2021.



帕特里克·

埃塞、罗宾

·龙巴赫和

比约恩·奥

默。驯服高

分辨率图像

合成的变压

器。202

1



年在 CV

PR。



Ruiyu

an Ga

o, Ch

enche

n Zha

o, La

nqing

 Hong

, and

 Qian

g Xu.

 Diff

Guard

: Sem

antic

 mism

atchg

uided

 out-

of-di

strib

ution

 dete

ction

 usin

g pre

-trai

ned d

iffus

ion m

odels

. In 

ICCV,

 2023

.



高瑞元，，

赵，洪蓝青

，。Dif

fGuar

d:使用预

训练扩散模

型的语义不

匹配引导的

分布外检



测。在 I

CCV，2

023 年

。



Chong

jian 

Ge, J

unson

g Che

n, En

ze Xi

e, Zh

ongda

o Wan

g, La

nqing

 Hong

, Huc

huan 

Lu, Z

hengu

o



Li, a

nd Pi

ng Lu

o. Me

taBEV

: Sol

ving 

senso

r fai

lures

 for 

bev d

etect

ion a

nd ma

p seg

menta

tion.



In IC

CV, 2

023.



葛崇建，，

陈，谢恩泽

，王中道，

洪蓝青，陆

沪川，，罗

平。Met

aBEV:

解决 BE

V 检测和

地



图分割的传

感器故障。

在 ICC

V，202

3 年。



Yunha

o Gou

, Zhi

li Li

u, Ka

i Che

n, La

nqing

 Hong

, Han

g Xu,

 Aoxu

e Li,

 Dit-

Yan Y

eung,

 Jame

s T



Kwok,

 and 

Yu Zh

ang. 

Mixtu

re of

 clus

ter-c

ondit

ional

 lora

 expe

rts f

or vi

sion-

langu

age



instr

uctio

n tun

ing. 

arXiv

 prep

rint 

arXiv

:2312

.1237

9, 20

23.



苟，，刘志

立，，洪蓝

青，，李敖

雪，杨迪燕

，郭展灏和

。视觉语言

指令调谐的

群集条件



lora 

专家的混合

。arXi

v 预印本

 arXi

v:231

2.123

79，20

23。



Nicho

las G

utten

berg.

 Diff

usion

 with

 offs

et no

ise. 

https

://ww

w.cro

sslab

s.org

/blog

/



diffu

sion-

with-

offse

t-noi

se, 2

023.



尼 古 拉

 斯 · 

古 滕 贝

 格 。 

带 有 偏

 移 噪 

声 的 扩

 散 。 

https

://ww

w.cro

sslab

s.org

/blog

/



diffu

sion-

with-

offse

t-noi

se, 2

023.



Jianh

ua Ha

n, Xi

wen L

iang,

 Hang

 Xu, 

Kai C

hen, 

Lanqi

ng Ho

ng, C

haoqi

ang Y

e, We

i Zha

ng, Z

hengu

o Li,

 Xiao

dan L

iang,

 and 

Chunj

ing X

u. So

da10m

: Tow

ards 

large

-scal

e obj

ect d

etect

ion



bench

- mar

k for

 auto

nomou

s dri

ving.

 arXi

v pre

print

 arXi

v:210

6.111

18, 2

021.



、梁、、、

洪蓝青、叶

超强、、、

梁和徐春净

。Soda

10m:走

向自动驾驶

的大规模目

标检测



基准。ar

Xiv 预

印本 ar

Xiv:2

106.1

1118，

2021。



Jonat

han H

o and

 Tim 

Salim

ans. 

Class

ifier

-free

 diff

usion

 guid

ance.

 In N

eurIP

S 202

1 Wor

kshop

 on



Deep 

Gener

ative

 Mode

ls an

d Dow

nstre

am Ap

plica

tions

, 202

1.



乔纳森·何

和蒂姆·萨

利曼斯。无

分类器扩散

制导。在 

NeurI

PS 20

21 深度

生成模型和

下游



应用研讨会

上，202

1。



Jonat

han H

o, Aj

ay Ja

in, a

nd Pi

eter 

Abbee

l. De

noisi

ng di

ffusi

on pr

obabi

listi

c mod

els. 

In Ne

urIPS

,



2020.



乔纳森·何

，阿贾伊·

贾恩和彼得

·阿比勒。

去噪扩散概

率模型。在

 Neur

IPS，2

020 年

。



Junji

e Hua

ng, G

uan H

uang,

 Zhen

g Zhu

, Ye 

Yun, 

and D

along

 Du. 

Bevde

t: Hi

gh-pe

rform

ance



multi

- cam

era 3

d obj

ect d

etect

ion i

n bir

d-eye

-view

. arX

iv pr

eprin

t arX

iv:21

12.11

790, 

2021.



黄俊捷、黄

冠、郑竹、

叶韵和杜大

龙。Bev

det:高

性能多摄像

机鸟瞰三维

物体检测。

arXiv



预印本 a

rXiv:

2112.

11790

，2021

。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



Yuanf

eng J

i, Zh

e Che

n, En

ze Xi

e Xie

, Lan

qing 

Hong,

 Xihu

i Liu

, Zha

oqian

g Liu

, Ton

g Lu,



Zheng

uo Li

, and

 Ping

 Luo.

 DDP:

 Diff

usion

 mode

l for

 dens

e vis

ual p

redic

tion.

 In I

CCV, 

2023.



纪元峰，，

谢恩泽，洪

蓝青，，刘

，，，，罗

平。DDP

:用于密集

视觉预测的

扩散模型。

在



ICCV，

2023 

年。



Kaica

n Li,

 Kai 

Chen,

 Haoy

u Wan

g, La

nqing

 Hong

, Cha

oqian

g Ye,

 Jian

hua H

an, Y

ukuai

 Chen

,



Wei Z

hang,

 Chun

jing 

Xu, D

it-Ya

n Yeu

ng, e

t al.

 Coda

: A r

eal-w

orld 

road 

corne

r cas

e dat

aset 

for



objec

t det

ectio

n in 

auton

omous

 driv

ing. 

arXiv

 prep

rint 

arXiv

:2203

.0772

4, 20

22.



李开灿，，

，洪蓝青，

叶超强，，

陈玉快，，

，杨迪燕，

等。Cod

a:一个用

于自动驾驶

中物



体检测的真

实道路拐角

数据集。a

rXiv 

预印本 a

rXiv:

2203.

07724

，2022

。



Pengx

iang 

Li, Z

hili 

Liu, 

Kai C

hen, 

Lanqi

ng Ho

ng, Y

unzhi

 Zhug

e, Di

t-Yan

 Yeun

g, Hu

chuan

 Lu,



and X

u Jia

. Tra

ckdif

fusio

n: Mu

lti-o

bject

 trac

king 

data 

gener

ation

 via 

diffu

sion 

model

s. ar

Xiv



prepr

int a

rXiv:

2312.

00651

, 202

3a.



、李、刘至

立、、洪蓝

青、诸葛、

杨迪燕、陆

沪川、。跟

踪扩散:通

过扩散模型

的多目标跟



踪数据生成

。arXi

v 预印本

 arXi

v:231

2.006

51，20

23a。



Yuhen

g Li,

 Haot

ian L

iu, Q

ingya

ng Wu

, Fan

gzhou

 Mu, 

Jianw

ei Ya

ng, J

ianfe

ng Ga

o, Ch

unyua

n Li,



and Y

ong J

ae Le

e. Gl

igen:

 Open

-set 

groun

ded t

ext-t

o-ima

ge ge

nerat

ion. 

In CV

PR, 2

023b.



李宇恒、刘

昊天、吴青

阳、穆、、

高剑锋、李

春元和李勇

在。Gli

gen:基

于开放集的

文本到



图像生成。

在 CVP

R，202

3 年。



Tsung

-Yi L

in, M

ichae

l Mai

re, S

erge 

Belon

gie, 

James

 Hays

, Pie

tro P

erona

, Dev

a Ram

anan,

 Piot

r



Dolla

´r, a

nd C 

Lawre

nce Z

itnic

k. Mi

croso

ft co

co: C

ommon

 obje

cts i

n con

text.

 In E

CCV, 

2014.



宗-林逸、

迈克尔·梅

尔、塞尔日

·贝隆吉、

詹姆斯·海

斯、彼得罗

·佩罗娜、

迪瓦·拉马



南、彼得·

多拉·r 

和 C·劳

伦斯·兹尼

克。微软 

coco:

上下文中的

公共对象。

2014 

年在



ECCV。



Nan L

iu, S

huang

 Li, 

Yilun

 Du, 

Anton

io To

rralb

a, an

d Jos

hua B

 Tene

nbaum

. Com

posit

ional



visua

l gen

erati

on wi

th co

mposa

ble d

iffus

ion m

odels

. In 

ECCV,

 2022

a.



刘楠，，杜

宜伦，An

tonio

 Torr

alba 

和 Jos

hua B

 Tene

nbaum

。用可组合

扩散模型合

成视觉



生成。在 

ECCV，

2022 

年。



Ze Li

u, Yu

tong 

Lin, 

Yue C

ao, H

an Hu

, Yix

uan W

ei, Z

heng 

Zhang

, Ste

phen 

Lin, 

and B

ainin

g



Guo. 

Swin 

trans

forme

r: Hi

erarc

hical

 visi

on tr

ansfo

rmer 

using

 shif

ted w

indow

s. In

 ICCV

, 202

1.



、林语桐、

、韩虎、、

魏、、林和

郭柏宁。S

win t

ransf

ormer

:使用移位

窗口的分层

视觉转



换器。20

21 年在

 ICCV

。



Zhiji

an Li

u, Ha

otian

 Tang

, Ale

xande

r Ami

ni, X

ingyu

 Yang

, Hui

zi Ma

o, Da

niela

 Rus,

 and 

Song



Han. 

Bevfu

sion:

 Mult

i-tas

k mul

ti-se

nsor 

fusio

n wit

h uni

fied 

bird’

s-eye

 view

 repr

esent

ation

. In



ICRA,

 2023

a.



、唐浩天、

亚力山大阿

米尼、杨兴

宇、毛、丹

妮拉鲁斯和

宋寒。Be

vfusi

on:具有

统一鸟瞰



视图表示的

多任务多传

感器融合。

2023 

年在 IC

RA。



Zhili

 Liu,

 Jian

hua H

an, K

ai Ch

en, L

anqin

g Hon

g, Ha

ng Xu

, Chu

njing

 Xu, 

and Z

hengu

o Li.

 Task

custo

mized

 self

-supe

rvise

d pre

-trai

ning 

with 

scala

ble d

ynami

c rou

ting.

 In A

AAI, 

2022b

.



刘志立、、

、洪蓝青、

、和。具有

可扩展动态

路由的任务

定制的自我

监督预训练

。在



AAAI，

2022b

。



Zhili

 Liu,

 Kai 

Chen,

 Yifa

n Zha

ng, J

ianhu

a Han

, Lan

qing 

Hong,

 Hang

 Xu, 

Zheng

uo Li

, Dit

-Yan



Ye- u

ng, a

nd Ja

mes K

wok. 

Geom-

erasi

ng: G

eomet

ry-dr

iven 

remov

al of

 impl

icit 

conce

pt in



diffu

sion 

model

s. ar

Xiv p

repri

nt ar

Xiv:2

310.0

5873,

 2023

b.



刘志立、、

、、洪蓝青

、、、董伟

强和郭炳湘

。几何删除

:扩散模式

中隐含概念

的几何驱动



删除。ar

Xiv 预

印本 ar

Xiv:2

310.0

5873，

2023b

。



Ilya 

Loshc

hilov

 and 

Frank

 Hutt

er. D

ecoup

led w

eight

 deca

y reg

ulari

zatio

n. In

 ICLR

, 201

9.



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



伊利亚·洛

希洛夫和弗

兰克·哈特

。去耦权重

衰减正则化

。在 IC

LR，20

19 年。



Ben M

ilden

hall,

 Prat

ul P 

Srini

vasan

, Mat

thew 

Tanci

k, Jo

natha

n T B

arron

, Rav

i Ram

amoor

thi, 

and



Ren N

g. Ne

rf: R

epres

entin

g sce

nes a

s neu

ral r

adian

ce fi

elds 

for v

iew s

ynthe

sis. 

In EC

CV, 2

020.



本·米尔登

霍尔、普拉

图尔·P·

斯里尼瓦桑

、马修·坦

西克、乔纳

森·T·巴

伦、拉维·



拉马穆尔蒂

和 Ren

 Ng。N

erf:将

场景表示为

用于视图合

成的神经辐

射场。在 

2020 

年的



ECCV。



Alexa

nder 

Quinn

 Nich

ol, P

raful

la Dh

ariwa

l, Ad

itya 

Rames

h, Pr

anav 

Shyam

, Pam

ela M

ishki

n,



Bob M

cgrew

, Ily

a Sut

skeve

r, an

d Mar

k Che

n. Gl

ide: 

Towar

ds ph

otore

alist

ic im

age g

enera

tion



and e

ditin

g wit

h tex

t-gui

ded d

iffus

ion m

odels

. In 

ICML,

 2022

.



亚历山大·

奎因·尼科

尔、普拉富

拉·达里瓦

尔、阿迪蒂

亚·拉梅什

、普拉纳夫

·希亚



姆、帕梅拉

·米什金、

鲍勃·麦克

格鲁、伊利

亚·苏茨基

弗和陈唐山

。Glid

e:使用文

本



引导扩散模

型实现照片

级真实感图

像生成和编

辑。在 I

CML，2

022 年

。



Alec 

Radfo

rd, J

ong W

ook K

im, C

hris 

Halla

cy, A

ditya

 Rame

sh, G

abrie

l Goh

, San

dhini

 Agar

wal,



Giris

h Sas

try, 

Amand

a Ask

ell, 

Pamel

a Mis

hkin,

 Jack

 Clar

k, et

 al. 

Learn

ing t

ransf

erabl

e vis

ual



model

s fro

m nat

ural 

langu

age s

uperv

ision

. In 

ICML,

 2021

.



亚历克·拉

德福德、琼

·金旭、克

里斯·哈拉

西、阿迪蒂

亚·拉梅什

、加布里埃

尔·高、



桑蒂尼·阿

加瓦尔、吉

里什·萨斯

特里、阿曼

达·阿斯克

尔、帕梅拉

·米什金、

杰克·



克拉克等人

,《从自然

语言监督中

学习可转移

视觉模型》

。2021

 年在 I

CML。



Robin

 Romb

ach, 

Andre

as Bl

attma

nn, D

omini

k Lor

enz, 

Patri

ck Es

ser, 

and B

jo¨rn

 Omme

r. Hi

ghres

oluti

on im

age s

ynthe

sis w

ith l

atent

 diff

usion

 mode

ls. I

n CVP

R, 20

22.



罗宾·龙巴

赫、安德里

亚斯·布拉

特曼、张秀

坤·洛伦茨

、帕特里克

·埃塞尔和

比约



·rn·奥

默。用潜在

扩散模型合

成高分辨率

图像。在 

CVPR，

2022 

年。



Yang 

Song,

 Jasc

ha So

hl-Di

ckste

in, D

ieder

ik P 

Kingm

a, Ab

hishe

k Kum

ar, S

tefan

o Erm

on, a

nd



Ben P

oole.

 Scor

e-bas

ed ge

nerat

ive m

odeli

ng th

rough

 stoc

hasti

c dif

feren

tial 

equat

ions.

 In I

CLR,



2020.



宋洋、贾沙

·苏尔-迪

克斯坦、迪

德里克·金

马、阿布舍

克·库马尔

、斯特凡诺

·埃尔蒙和



本·普尔。

基于分数的

随机微分方

程生成模型

。在 20

20 年的

 ICLR

。



Alexa

nder 

Swerd

low, 

Runsh

eng X

u, an

d Bol

ei Zh

ou. S

treet

-view

 imag

e gen

erati

on fr

om a 

bird’

seye 

view 

layou

t. ar

Xiv p

repri

nt ar

Xiv:2

301.0

4634,

 2023

.



Alexa

nder 

Swerd

low 、

 徐 润 

生 和 周

 。 从 

鸟 瞰 图

 布 局 

生 成 街

 景 图 

像 。 a

rXiv 

预 印 本



arXiv

:2301

.0463

4，202

3。



Shita

o Tan

g, Fu

yang 

Zhang

, Jia

cheng

 Chen

, Pen

g Wan

g, an

d Yas

utaka

 Furu

kawa.

 Mvdi

ffusi

on:



En- a

bling

 holi

stic 

multi

-view

 imag

e gen

erati

on wi

th co

rresp

onden

ce-aw

are d

iffus

ion. 

arXiv



prepr

int a

rXiv:

2307.

01097

, 202

3.



唐世涛，张

富阳，陈家

成，，古川

。Mvdi

ffusi

on:利用

一致性感知

扩散实现整

体多视图图



像生成。a

rXiv 

预印本 a

rXiv:

2307.

01097

，2023

。



Hung-

Yu Ts

eng, 

Qinbo

 Li, 

Chang

il Ki

m, Su

hib A

lsisa

n, Ji

a-Bin

 Huan

g, an

d Joh

annes

 Kopf

. Con

siste

nt vi

ew sy

nthes

is wi

th po

se-gu

ided 

diffu

sion 

model

s. In

 CVPR

, 202

3.



hong-

Yu Ts

eng，Q

inbo 

Li，Ch

angil

 Kim，

Suhib

 Alsi

san，J

ia-黄斌

和 Joh

annes

 Kopf

。



利用姿态引

导扩散模型

的一致视图

合成。在 

CVPR，

2023 

年。



Ashis

h Vas

wani,

 Noam

 Shaz

eer, 

Niki 

Parma

r, Ja

kob U

szkor

eit, 

Llion

 Jone

s, Ai

dan N

 Gome

z,



Łukas

z Kai

ser, 

and I

llia 

Polos

ukhin

. Att

entio

n is 

all y

ou ne

ed. I

n Neu

rIPS,

 2017

.



Ashis

h Vas

wani、

Noam 

Shaze

er、Ni

ki Pa

rmar、

Jakob

 Uszk

oreit

、Llio

n Jon

es、Ai

dan



Gomez

、ukas

z Kai

ser 和

 Illi

a Pol

osukh

in。你需

要的只是关

注。在 N

eurIP

S，201

7。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



Su Wa

ng, C

hitwa

n Sah

aria,

 Cesl

ee Mo

ntgom

ery, 

Jordi

 Pont

-Tuse

t, Sh

ai No

y, St

efano

 Pell

egrin

i,



Yasum

asa O

noe, 

Sarah

 Lasz

lo, D

avid 

J Fle

et, R

adu S

oricu

t, et

 al. 

Image

n edi

tor a

nd ed

itben

ch:



Advan

cing 

and e

valua

ting 

text-

guide

d ima

ge in

paint

ing. 

In CV

PR, 2

023a.



王素，奇万

·萨哈利亚

，塞斯利·

蒙哥马利，

乔迪·庞特

-图塞特，

沙伊·诺伊

，斯特凡诺



·佩莱格里

尼，小野保

正，萨拉·

拉兹洛，戴

维·J·弗

利特，拉杜

·索里科特

，



等。202

3 年在 

CVPR。



Weilu

n Wan

g, Ji

anmin

 Bao,

 Weng

ang Z

hou, 

Dongd

ong C

hen, 

Dong 

Chen,

 Lu Y

uan, 

and



Houqi

ang L

i. Se

manti

c ima

ge sy

nthes

is vi

a dif

fusio

n mod

els. 

arXiv

 prep

rint



arXiv

:2207

.0005

0, 20

22.



、鲍建民、

周文刚、、

、陆源和李

。基于扩散

模型的语义

图像合成。

 arXi

v 预印本



arXiv

:2207

.0005

0，202

2。



Xiaof

eng W

ang, 

Zheng

 Zhu,

 Guan

 Huan

g, Xi

nze C

hen, 

Jiaga

ng Zh

u, an

d Jiw

en Lu

. Dri

vedre

amer:

 Towa

rds r

eal-w

orld-

drive

n wor

ld mo

dels 

for a

utono

mous 

drivi

ng. a

rXiv 

prepr

int



arXiv

:2309

.0977

7, 20

23b.



、郑竹、、

陈新泽、朱

家刚和陆继

文。驾驶梦

想家:走向

现实世界的

自动驾驶世

界模



型。arX

iv 预印

本 arX

iv:23

09.09

777，2

023b。



Xiaof

eng W

ang, 

Zheng

 Zhu,

 Yunp

eng Z

hang,

 Guan

 Huan

g, Yu

n Ye,

 Wenb

o Xu,

 Ziwe

i Che

n, an

d



Xinga

ng Wa

ng. A

re we

 read

y for

 visi

on-ce

ntric

 driv

ing s

tream

ing p

ercep

tion?

 the 

asap 

bench

mark.

 In C

VPR, 

2023c

.



、郑竹、、

、、、陈、

和王心刚。

我们准备好

迎接以视觉

为中心的驱

动流感知了

吗？尽快



基准。在 

CVPR，

2023c

。



Jay Z

hangj

ie Wu

, Yix

iao G

e, Xi

ntao 

Wang,

 Stan

 Weix

ian L

ei, Y

uchao

 Gu, 

Yufei

 Shi,

 Wynn

e



Hsu, 

Ying 

Shan,

 Xiao

hu Qi

e, an

d Mik

e Zhe

ng Sh

ou. T

une-a

-vide

o: On

e-sho

t tun

ing o

f ima

ge



diffu

sion 

model

s for

 text

-to-v

ideo 

gener

ation

. In 

CVPR,

 2023

a.



周杰伦吴，

葛，王，雷

伟贤，顾，

史，许永利

，，肖虎和

郑守迈。视

频调谐:文

本到视频生



成的图像扩

散模型的一

次性调谐。

2023 

年在 CV

PR。



Weiji

a Wu,

 Yuzh

ong Z

hao, 

Hao C

hen, 

Yucha

o Gu,

 Rui 

Zhao,

 Yefe

i He,

 Hong

 Zhou

, Mik

e Zhe

ng



Shou,

 and 

Chunh

ua Sh

en. D

atase

tdm: 

Synth

esizi

ng da

ta wi

th pe

rcept

ion a

nnota

tions

 usin

g



diffu

- sio

n mod

els. 

arXiv

 prep

rint 

arXiv

:2308

.0616

0, 20

23b.



、赵玉忠、

、、顾、、

、、、郑守

迈和。Da

taset

dm:使用

扩散模型合

成带有感知

注释的数



据。arX

iv 预印

本 arX

iv:23

08.06

160，2

023b。



Kairu

i Yan

g, En

hui M

a, Ji

bin P

eng, 

Qing 

Guo, 

Di Li

n, an

d Kai

cheng

 Yu. 

Bevco

ntrol

: Acc

urate

ly



contr

ollin

g str

eet-v

iew e

lemen

ts wi

th mu

lti-p

erspe

ctive

 cons

isten

cy vi

a bev

 sket

ch la

yout.



arXiv

 prep

rint 

arXiv

:2308

.0166

1, 20

23a.



杨，马恩慧

，，彭，，

郭庆，，俞

开成。Be

vcont

rol:通

过 bev

 草图布局

，精确控制

多视角



一致性的街

景元素。a

rXiv 

预印本 a

rXiv:

2308.

01661

，2023

a。



Yijun

 Yang

, Rui

yuan 

Gao, 

Xiaos

en Wa

ng, N

an Xu

, and

 Qian

g Xu.

 Mma-

diffu

sion:

 Mult

imoda

l



attac

k on 

diffu

sion 

model

s. ar

Xiv p

repri

nt ar

Xiv:2

311.1

7516,

 2023

b.



、高瑞元、

王小森、徐

楠和徐强。

MMA-扩

散:对扩散

模型的多模

态攻击。a

rXiv 

预印本



arXiv

:2311

.1751

6，202

3b。



Lvmin

 Zhan

g, An

yi Ra

o, an

d Man

eesh 

Agraw

ala. 

Addin

g con

ditio

nal c

ontro

l to 

text-

to-im

age



diffu

sion 

model

s. In

 ICCV

, 202

3a.



张、饶安怡

和马涅什·

阿格拉瓦拉

。向文本到

图像扩散模

型添加条件

控制。20

23 年在



ICCV。



Shu Z

hang,

 Xiny

i Yan

g, Yi

hao F

eng, 

Can Q

in, C

hia-C

hih C

hen, 

Ning 

Yu, Z

eyuan

 Chen

, Hua

n



Wang,

 Silv

io Sa

vares

e, St

efano

 Ermo

n, et

 al. 

Hive:

 Harn

essin

g hum

an fe

edbac

k for



instr

uctio

nal v

isual

 edit

ing. 

arXiv

 prep

rint 

arXiv

:2303

.0961

8, 20

23b.



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



、杨欣怡、

冯一豪、秦

灿灿、陈嘉

芝、、陈、

、西尔维奥

·萨瓦雷塞

、斯特凡诺

·埃蒙等



编著。ar

Xiv 预

印本 ar

Xiv:2

303.0

9618，

2023b

。



Wenli

ang Z

hao, 

Lujia

 Bai,

 Yong

ming 

Rao, 

Jie Z

hou, 

and J

iwen 

Lu. U

nipc:

 A un

ified

 pred

ictor

corre

ctor 

frame

work 

for f

ast s

ampli

ng of

 diff

usion

 mode

ls. a

rXiv 

prepr

int a

rXiv:

2302.

04867

,



2023.



赵，白路佳

，饶永明，

周杰，陆继

文。Uni

pc:扩散

模式快速取

样的统一预

测-校正框



架。arX

iv 预印

本 arX

iv:23

02.04

867，2

023。



Ziyan

g Zhe

ng, R

uiyua

n Gao

, and

 Qian

g Xu.

 Non-

cross

 diff

usion

 for 

seman

tic c

onsis

tency

. arX

iv



prepr

int a

rXiv:

2312.

00820

, 202

3.



郑子扬，高

瑞元，。语

义一致性的

非交叉扩散

。arXi

v 预印本

 arXi

v:231

2.008

20，20

23。



LIU Z

hili,

 Kai 

Chen,

 Jian

hua H

an, H

ONG L

anqin

g, Ha

ng Xu

, Zhe

nguo 

Li, a

nd Ja

mes K

wok.



Task-

 cust

omize

d mas

ked a

utoen

coder

 via 

mixtu

re of

 clus

ter-c

ondit

ional

 expe

rts. 

In IC

LR,



2023.



刘志立、程

凯、韩建华

、洪蓝青、

徐航、李振

国和郭炳湘

。通过混合

聚类条件专

家的任务



定制屏蔽自

动编码器。

在 ICL

R，202

3 年。



Bolei

 Zhou

, Han

g Zha

o, Xa

vier 

Puig,

 Tete

 Xiao

, San

ja Fi

dler,

 Adel

a Bar

riuso

, and

 Anto

nio



Torra

lba. 

Seman

tic u

nders

tandi

ng of

 scen

es th

rough

 the 

ade20

k dat

aset.

 In I

JCV, 

2019.



周、、泽维

尔·普伊格

、肖太特、

萨尼亚·菲

德勒、·巴

里乌索和安

东尼奥·托

雷巴。通



过 ade

20k 数

据集对场景

进行语义理

解。在 I

JCV，2

019 年

。



Brady

 Zhou

 and 

Phili

pp Kr

a¨hen

bu¨hl

. Cro

ss-vi

ew tr

ansfo

rmers

 for 

real-

time 

map-v

iew s

emant

ic



segme

ntati

on. I

n CVP

R, 20

22.



布雷迪周和

菲利普克拉

亨布 hl

。用于实时

地图视图语

义分割的交

叉视图转换

器。在



CVPR，

2022 

年。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



APPEN

DIX



附录



A OBJ

ECT F

ILTER

ING



B 对象过

滤



(a) A

ll ob

ject 

boxes

 (b) 

visib

le bo

xes f

or Fr

ont L

eft v

iew (

c) vi

sible

 boxe

s for

 Back

 Righ

t vie

w



(a)所有

目标框 (

b)左前视

图的可视框

(c)右后

视图的可视

框



Figur

e 10:

 Boun

ding 

box f

ilter

ing t

o eac

h vie

w. Th

e das

hed l

ines 

in (b

-c) r

epres

ent t

he x-

axis 

of



the c

amera

’s co

ordin

ates.

 Boxe

s are

 reta

ined 

only 

if th

ey ha

ve at

 leas

t a p

oint 

in th

e pos

itive

 half



of th

e z-a

xis i

n eac

h cam

era’s

 coor

dinat

e.



图 10:

每个视图的

边界框过滤

。(b-c

)中的虚线

表示相机坐

标的 x 

轴。仅当长

方体在每个

相



机坐标的 

z 轴正半

部分至少有

一个点时，

才会保留长

方体。



In Eq

uatio

n 6, 

we em

ploy 

fviz 

for o

bject

 filt

ering

 to f

acili

tate 

boots

trap 

learn

ing. 

We sh

ow



more 

detai

ls of

 fviz

 here

. Ref

er to

 Figu

re 10

 for 

illus

trati

on. F

or th

e sak

e of 

simpl

icity

, eac

h



camer

a’s F

ield 

Of Vi

ew (F

OV) i

s not

 cons

idere

d. Ob

jects

 are 

defin

ed as

 visi

ble i

f any

 corn

er of



their

 boun

ding 

boxes

 is l

ocate

d in 

front

 of t

he ca

mera 

(i.e.

, z



vi > 

0) wi

thin 

each 

camer

a’s



coord

inate

 syst

em. T

he ap

plica

tion 

of fv

iz si

gnifi

cantl

y lig

htens

 the 

workl

oad o

f the

 boun

ding



box e

ncode

r, ev

idenc

e for

 whic

h can

 be f

ound 

in Se

ction

 6.



在等式中 

6，我们采

用 fvi

z 进行对

象过滤，以

促进引导学

习。我们在

这里展示了

 fviz



的更多细节

。参考图 

10 为了

说明。为了

简单起见，

不考虑每个

摄像机的视

野(FOV

)。如果



对象的边界

框的任何角

在每个摄像

机的坐标系

内位于摄像

机的前面(

即 zvi

 > 0 

),则对象



被定义为可

见。fvi

z 的应用

大大减轻了

包围盒编码

器的工作量

，其证据可

以在第节中

找到



6。



C MOR

E EXP

ERIME

NTAL 

DETAI

LS



D 更多实

验细节



Seman

tic C

lasse

s for

 Gene

ratio

n. To

 supp

ort m

ost p

ercep

tion 

model

s on 

nuSce

nes, 

we tr

y to



inclu

de se

manti

cs co

mmonl

y use

d in 

most 

setti

ngs (

Huang

 et a

l., 2

021; 

Zhou 

& Kra

¨henb

u¨hl,



2022;

 Liu 

et al

., 20

23a; 

Ge et

 al.,

 2023

). Sp

ecifi

cally

, for

 obje

cts, 

ten c

atego

ries 

inclu

de ca

r, bu

s,



truck

, tra

iler,

 moto

rcycl

e, bi

cycle

, con

struc

tion 

vehic

le, p

edest

rian,

 barr

ier, 

and t

raffi

c con

e. Fo

r



the r

oad m

ap, e

ight 

categ

ories

 incl

ude d

rivab

le ar

ea, p

edest

rian 

cross

ing, 

walkw

ay, s

top l

ine, 

car



parki

ng ar

ea, r

oad d

ivide

r, la

ne di

vider

, and

 road

block

.



用于生成的

语义类。为

了支持 n

uScen

es 上的

大多数感知

模型，我们

试图包含大

多数设置中



常用的语义

(Huan

g et 

al.,2

021；Z

hou &

 Kra¨

henbu

¨hl,2

022；L

iu et

 al.,

2023a

；Ge



et al

.,202

3).具体

来说，对于

对象，十个

类别包括汽

车、公共汽

车、卡车、

拖车、摩托



车、自行车

、建筑车辆

、行人、障

碍物和交通

锥。对于道

路地图，八

个类别包括

可行驶区



域、人行横

道、人行道

、停车线、

停车场、道

路分隔带、

车道分隔带

和路障。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



Optim

izati

on. W

e tra

in al

l new

ly ad

ded p

arame

ters 

using

 Adam

W (Lo

shchi

lov &

 Hutt

er, 2

019)



optim

izer 

and a

 cons

tant 

learn

ing r

ate a

t 8e



−5



and b

atch 

size 

24 (t

otal 

144 i

mages

 for 

6 vie

ws)



with 

a lin

ear w

arm-u

p of 

3000 

itera

tions

, and

 set 

γ



s = 0

.2.



优化。我们

使用 Ad

amW(L

oshch

ilov 

& Hut

ter,2

019)优

化器和 8

e 5 下

的恒定学习

速率，批



量大小为 

24(6 

个视图总共

 144 

个图像)，

线性预热 

3000 

次迭代，设

置 γs 

= 0.2

。



E ABL

ATION

 ON N

UMBER

 OF A

TTEND

ING V

IEWS



F 根据参

与视图数量

进行消融



Table

 5: A

blati

on on

 numb

er of

 atte

nding

 view

s. Ev

aluat

ion r

esult

s are

 from

 CVT 

on th

e syn

theti

c



nuSce

nes v

alida

ti on

 set,

 with

out M

 = { 

0 } i

n CFG

 scal

e = 2

.



表 5:参

与视图数量

上的消融。

评估结果来

自合成 n

uScen

es 验证

集上的 C

VT，在 

CFG s

cale



= 2 中

没有 M 

= {0}

。



 Atte

nding

 Num.

 FID 

↓ Roa

d mIo

U ↑ V

ehicl

e mIo

U ↑



 主治编号

。 FID

 ↓ 米欧

路 车辆 

mIoU



1 13.

06 58

.63 2

6.41



一 13.

06 58

.63 2

6.41



2 (ou

rs) 1

4.46 

59.31

 27.1

3



2(我们的

) 14.

46 59

.31 2

7.13



5 (al

l) 14

.76 5

8.35 

25.41



5(全部)

 14.7

6 58.

35 25

.41



In Ta

ble 5

, we 

demon

strat

e the

 impa

ct of

 vary

ing t

he nu

mber 

of at

tende

d vie

ws on

 eval

uatio

n res

ults.

 Atte

nding

 to a

 sing

le vi

ew yi

elds 

super

ior F

ID re

sults

; the

 redu

ced i

nflux

 of i

nform

ation



from 

neigh

borin

g vie

ws si

mplif

ies o

ptimi

zatio

n for

 that

 view

. How

ever,

 this

 appr

oach



compr

omise

s mIo

U, al

so re

flect

ing l

ess c

onsis

tent 

gener

ation

, as 

depic

ted i

n Fig

ure 1

1.



Conve

rsely

, inc

orpor

at- i

ng al

l vie

ws de

terio

rates

 perf

orman

ce ac

ross 

all m

etric

s, po

tenti

ally 

due



to ex

cessi

ve in

forma

tion 

causi

ng in

terfe

rence

 in c

ross-

atten

tion.

 Sinc

e eac

h vie

w has

 an



inter

secti

on wi

th bo

th le

ft an

d rig

ht vi

ews, 

atten

ding 

to on

e vie

w can

not g

uaran

tee c

onsis

tency

,



espec

ially

 for 

foreg

round

 obje

cts, 

while

 atte

nding

 to m

ore v

iews 

requi

res m

ore c

omput

ation

. Thu

s,



we op

t for

 2 at

tende

d vie

ws in

 our 

main 

paper

, str

iking

 a ba

lance

 betw

een c

onsis

tency

 and



compu

tatio

nal e

ffici

ency.



在表中 5

 中，我们

展示了改变

参与视图的

数量对评估

结果的影响

。关注单一

视图会产生

更



好的 FI

D 结果；

来自相邻视

图的减少的

信息流入简

化了该视图

的优化。然

而，这种方

法损



害了 mI

oU，也反

映了不太一

致的生成，

如图所示 

11。相反

，合并所有

视图会降低

所有指标



的性能，这

可能是因为

过多的信息

会干扰交叉

关注。由于

每个视图与

左视图和右

视图都有



交集，所以

关注一个视

图不能保证

一致性，尤

其是对于前

景对象，而

关注更多的

视图需要



更多的计算

。因此，我

们在我们的

主论文中选

择了 2 

个参与视图

，在一致性

和计算效率

之



间取得了平

衡。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



Atten

d to 

2 vie

ws At

tend 

to 1 

view



关注 2 

个视图关注

 1 个视

图



Figur

e 11:

 Comp

ariso

n bet

ween 

diffe

rent 

numbe

rs of

 atte

nding

 view

s. On

ly at

tendi

ng to

 one 

view



resul

ts in

 wors

e mul

ti-ca

mera 

consi

stenc

y.



图 11:

不同数量的

参与视图之

间的比较。

仅关注一个

视图会导致

更差的多相

机一致性。



G QUA

LITAT

IVE C

OMPAR

ISON 

WITH 

BEVGE

N



H 与 B

EVGEN

 的定性比

较



Figur

e 12 

illus

trate

s tha

t MAG

ICDRI

VE ge

nerat

es im

ages 

with 

highe

r qua

lity 

compa

red t

o BEV

Gen (

Swerd

low e

t al.

, 202

3), p

artic

ularl

y exc

ellin

g in 

objec

ts. S

uch e

nhanc

ement

 can 

be at

tribu

ted



to MA

GICDR

IVE’s

 util

izati

on of

 the 

diffu

sion 

model

 and 

the a

dopti

on of

 a cu

stomi

zed c

ondit

ion



injec

tion 

strat

egy.



数字 12

 说明 M

AGICD

RIVE 

生成的图像

质量高于 

BEV- 

Gen(S

werdl

ow et

 al.,

2023)

，特别精



于物件。这

种增强可以

归功于 M

AGICD

RIVE 

对扩散模型

的利用和定

制条件注入

策略的采用

。



Fail 

to ge

n.



Low q

ualit

y



Wrong

 plac

e



未能生成。



低质量错误



位置



Groun

d Tru

th



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



Gener

ation

 from



Scene

 leve

l: pr

ompt 

+ “ni

ght, 

diffi

cult 

light

ing”



Backg

round

 leve

l: fl

ip ma

p and

 boxe

s fro

m lef

t to 

right

 (in 

world

 coor

dinat

e)



Gener

ation

 from



Scene

 leve

l: pr

ompt 

+ “ni

ght, 

diffi

cult 

light

ing”



Backg

round

 leve

l: fl

ip ma

p and

 boxe

s fro

m lef

t to 

right

 (in 

world

 coor

dinat

e)



地面实况;

 真值



（机器学习

）



BEVGe

n



贝夫根



Figur

e 12:

 Qual

itati

ve co

mpari

son w

ith B

EVGen

 on d

rivin

g sce

ne fr

om nu

Scene

s val

idati

on se

t.



We hi

ghlig

ht so

me ar

eas w

ith r

ectan

gles 

to ea

se co

mpari

son. 

Compa

red w

ith B

EVGen

, ima

ge



quali

ty of

 obje

cts f

rom M

AGICD

RIVE 

is mu

ch be

tter.



图 nuS

cenes

 验证集中

驾驶场景与

 BEVG

en 的定

性比较。为

了便于比较

，我们用矩

形突出显



示了一些区

域。与 B

EVGen

 相比，M

AGICD

RIVE 

的对象图像

质量要好得

多。



I MOR

E RES

ULTS 

WITH 

CONTR

OL FR

OM DI

FFERE

NT CO

NDITI

ONS



J 不同条

件下控制的

更多结果



Figur

e 13 

shows

 scen

e lev

el co

ntrol

 (tim

e of 

day) 

and b

ackgr

ound 

level

 cont

rol (

BEV m

ap al

terat

ions)

. MAG

ICDRI

VE ca

n eff

ectiv

ely r

eflec

t the

se ch

anges

 in c

ontro

l con

ditio

ns th

rough

 the 

gener

ated 

camer

a vie

ws.



数字 13

 显示场景

级别控制(

一天中的时

间)和背景

级别控制(

BEV 贴

图变更)。

MAGIC

DRIVE

 可



以通过生成

的摄像机视

图有效地反

映控制条件

的这些变化

。



Figur

e 13:

 Show

case 

for s

cene-

level

 cont

rol w

ith M

AGICD

RIVE.

 The 

scene

 is f

rom t

he nu

Scene

s



valid

ation

 set.



图 13:

使用 MA

GICDR

IVE 进

行场景级控

制的展示。

该场景来自

 nuSc

enes 

验证集。



mAP ↑

 NDS 

↑ mAP

 ↑



mAP ↑

 NDS 

↑ mAP

 ↑



×



× ×



×



× ×



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



K MOR

E EXP

ERIME

NTS W

ITH 3

D OBJ

ECT D

ETECT

ION



L 更多 

3D 物体

检测实验



Table

 6: C

ompar

ison 

about

 supp

ort f

or 3D

 obje

ct de

tecti

on mo

del (

i.e.,

 BEVF

usion

).



MAGIC

DRIVE

 gene

rates

 272×

736 i

mages

 for 

augme

ntati

on. R

esult

s are

 from

 test

s on 

the n

uScen

es



valid

ation

 set.



表 6:关

于支持 3

D 对象检

测模型(即

 BEVF

usion

)的比较。

MAGIC

DRIVE

 生成 2

72×73

6 的图像



用于增强。

结果来自对

 nuSc

enes 

验证集的测

试。



Epoch

 Data

 CAM-

Only 

CAM+L

iDAR



时代 数据

 仅限 C

AM 摄像

头+激光雷

达



NDS ↑



NDS ↑



0.5× 

w/o s

ynthe

tic d

ata



w/ MA

GICDR

IVE



30.21



33.29

 (+3.

08)



32.76



36.69

 (+3.

93) t

oo fe

w epo

chs t

oo fe

w epo

chs



0.5× 

无合成数据



带 MAG

ICDRI

VE



30.21



33.29



(+3.0

8)



32.76



36.69



(+3.9

3)



太少的时代

 太少的时

代



1× w/

o syn

theti

c dat

a



w/ MA

GICDR

IVE



32.88



35.40

 (+2.

52)



37.81



39.76

 (+1.

95)



65.40



67.86

 (+2.

46)



69.59



70.72

 (+1.

13)



1× 无合

成数据



带 MAG

ICDRI

VE



32.88



35.40



(+2.5

2)



37.81



39.76



(+1.9

5)



65.40



67.86



(+2.4

6)



69.59



70.72



(+1.1

3)



2× w/

o syn

theti

c dat

a



w/ MA

GICDR

IVE



35.49



35.74

 (+0.

25)



40.66



41.40

 (+0.

74)



68.33



68.58

 (+0.

25)



71.31



71.34

 (+0.

03)



2× 无合

成数据



带 MAG

ICDRI

VE



35.49



35.74



(+0.2

5)



40.66



41.40



(+0.7

4)



68.33



68.58



(+0.2

5)



71.31



71.34



(+0.0

3)



In Ta

ble 6

, we 

show 

addit

ional

 expe

rimen

tal r

esult

s on 

train

ing 3

D obj

ect d

etect

ion m

odels

 usin

g



synth

etic 

data 

produ

ced b

y MAG

ICDRI

VE. G

iven 

that 

BEVFu

sion 

utili

zes a

 ligh

tweig

ht ba

ckbon

e



(i.e.

, Swi

n-T (

Liu e

t al.

, 202

1)), 

model

 perf

orman

ce ap

pears

 to p

latea

u wit

h tra

ining

 thro

ugh 1

-



在表中 6

，我们展示

了使用 M

AGICD

RIVE 

生成的合成

数据训练 

3D 对象

检测模型的

附加实验结



果。假设 

BEVFu

sion 

利用了轻型

主干(即 

Swin-

T(Liu

 et a

l.,20

21))，

模型性能随

着通过



1-1 训

练而趋于平

稳



2 epo

chs (

2 : 2

0 for

 CAM-

Only 

and 6

 for 

CAM+L

iDAR)

. Red

ucing

 epoc

hs ca

n mit

igate

 this



satur

ation

, all

owing

 more

 vari

ed da

ta to

 enha

nce t

he mo

del’s

 perc

eptua

l cap

acity

 in b

oth s

ettin

gs.



This 

impro

vemen

t is 

evide

nt ev

en wh

en ep

ochs 

for 3

D obj

ect d

etect

ion a

re fu

rther

 redu

ced t

o 0.5



. Our

 MAGI

CDRIV

E acc

urate

ly au

gment

s str

eet-v

iew i

mages

 with

 the 

annot

ation

s. Fu

ture 

works



may f

ocus 

on an

notat

ion s

ampli

ng an

d con

struc

tion 

strat

egies

 for 

synth

etic 

data 

augme

ntati

on.



2 个历元

(仅 2 

: 20 

用于 CA

M，6 用

于 CAM

+激光雷达

)。减少历

元可以减轻

这种饱和度

，允



许更多不同

的数据来增

强模型在两

种设置下的

感知能力。

即使当 3

D 对象检

测的历元进

一步



减少到 0

.5 时，

这种改进也

是明显的。

我们的 M

AGICD

RIVE 

通过注释精

确地增强了

街景图



像。未来的

工作可能会

集中在注释

采样和合成

数据增强的

构造策略。



M MOR

E DIS

CUSSI

ON



N 更多讨

论



More 

futur

e wor

k. No

te th

at MA

GICDR

IVE-g

enera

ted s

treet

 view

s can

 curr

ently

 only

 perf

orm a

s



augme

nted 

sampl

es to

 trai

n wit

h rea

l dat

a, an

d it 

is ex

citin

g to 

train

 dete

ctors

 sole

ly wi

th ge

nerat

ed



data,

 whic

h wil

l be 

explo

red i

n the

 futu

re. M

ore f

lexib

le us

age o

f the

 gene

rated

 stre

et vi

ews



beyon

d dat

a aug

menta

tion,

 espe

ciall

y inc

orpor

ation

 with

 gene

rativ

e pre

-trai

ning 

(Chen

 et a

l.,



2023a

; Zhi

li et

 al.,

 2023

), co

ntras

tive 

learn

ing (

Chen 

et al

., 20

21; L

iu et

 al.,

 2022

b) an

d the

 larg

e



×



×



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



3



langu

age m

odels

 (LLM

s) (C

hen e

t al.

, 202

3b; G

ou et

 al.,

 2023

), is

 an a

ppeal

ing f

uture

 rese

arch



direc

tion.

 It i

s als

o int

erest

ing t

o uti

lize 

the g

eomet

ric c

ontro

ls in

 diff

erent

 circ

umsta

nces 

beyon

d



3D sc

enari

os (e

.g., 

multi

-obje

ct tr

ackin

g (Li

 et a

l., 2

023a)

 and 

conce

pt re

moval

 (Liu

 et a

l.,



2023b

)).



更多未来的

工作。请注

意，MAG

ICDRI

VE 生成

的街道视图

目前只能作

为增强样本

使用真实数



据进行训练

，仅使用生

成的数据训

练检测器是

令人兴奋的

，这将在未

来进行探索

。除了数



据扩充之外

，更灵活地

使用所生成

的街道视图

，尤其是与

生成性预训

练 (Ch

en et



al.,2

023a；

Zhili

et al

.,202

3)，对比

学习(Ch

en et

 al.,

2021；

Liu e

t al.

,2022

b)和大



型语言模型

(LLM)

(Chen

 et a

l.,20

23b；G

ou et

 al.,

2023)

，是一个很

有吸引力的

未来研究



方向。在 

3D 场景

之外的不同

环境中利用

几何控制也

是有趣的(

例如，多对

象跟踪(L

i et



al.,2

023a)

和概念移除

(Liu 

et al

.,202

3b)).



O DET

AILED

 ANAL

YSIS 

ON 3D

 OBJE

CT DE

TECTI

ON WI

TH SY

NTHET

IC DA

TA



P 利用合

成数据进行

三维目标检

测的详细分

析



Table

 7: P

er-cl

ass p

erfor

mance

 comp

ariso

n wit

h BEV

Fusio

n for

 3D o

bject

 dete

ction

 with

 1 se

tting

. Res

ults 

are t

ested

 on t

he nu

Scene

s val

idati

on se

t.



表 7:使

用 BEV

Fusio

n 在 1

 次设置下

进行 3D

 对象检测

时的每类性

能比较。结

果在 nu

Scene

s



验证集上进

行测试。



Data 

mAP c

ar co

ne ba

rrier

bus p

ed. m

otor.

 truc

k bic

ycle 

trail

er co

nstr.



数据 地图

 汽车锥形

路障。马达

。卡车自行

车拖车建筑

。



BEVFu

sion 

32.88

 50.6

7 50.

46 48

.62 3

7.73 

35.74

 30.4

0 27.

54 24

.85 1

5.56 

7.28



+ MAG

ICDRI

VE 35

.40 5

1.86 

53.56

 51.1

5 40.

43 38

.10 3

3.11 

29.35

 27.8

5 18.

74 9.

83



饮料融合 

32.8



8



50.67

 50.4

6 48.

6



2



37.73

 35.7



4



30.40

 27.5



4



24.8



5



15.5



6



7.2



8



+



Magic

Drive



35.4



0



51.86

 53.5

6 51.

1



5



40.43

 38.1



0



33.11

 29.3



5



27.8



5



18.7



4



9.8



3



Diffe

rence

 +2.5

2 +1.

20 +3

.10 +

2.53 

+2.70

 +2.3

6 +2.

71 +1

.81 +

3.00 

+3.19

 +2.5

5



差异 +2

.52 +

1.20 

+3.10

 +2.5

3 +2.

70 +2

.36 +

2.71 

+1.81

 +3.0

0 +3.

19 +2

.55



We pr

ovide

 per-

class

 AP f

or 3D

 obje

ct de

tecti

on fr

om th

e nuS

cenes

 vali

datio

n set

 usin

g



BEVFu

sion 

in Ta

ble 7

. Fro

m the

 resu

lts, 

we ob

serve

 that

, fir

stly,

 the 

impro

vemen

ts fo

r lar

ge



objec

ts ar

e sig

- nif

icant

, for

 exam

ple, 

buses

, tra

ilers

, and

 cons

truct

ion v

ehicl

es. S

econd

ly, o

bject

s



with 

less 

diver

se ap

peara

nces,

 such

 as t

raffi

c con

es an

d bar

riers

, sho

w mor

e imp

rovem

ent,



espec

ially

 comp

ared 

to tr

ucks.

 Thir

dly, 

we no

te th

at th

e imp

rovem

ent i

s mar

ginal

 for 

cars,

 whil

e



signi

fican

t for

 pede

stri-

 ans,

 moto

rcycl

es, a

nd bi

cycle

s. Th

is ma

y be 

becau

se th

e bas

eline

 alre

ady



perfo

rms w

ell f

or ca

rs. F

or pe

destr

ians,

 moto

rcycl

es, a

nd bi

cycle

s, ev

en th

ough 

dista

nt ob

jects



from 

the e

go ca

r are

 gen-

 erat

ed le

ss fa

ithfu

lly, 

MAGIC

DRIVE

 can 

synth

esize

 high

-qual

ity o

bject

s



near 

the e

go ca

r, as

 show

n in 

Figur

e 17-

18. T

heref

ore, 

more 

accur

ate d

etect

ion o

f obj

ects 

near 

the



ego c

ar co

ntrib

utes 

to im

- pro

vemen

ts fo

r the

se cl

asses

. Ove

rall,

 mAP 

impro

vemen

t com

es wi

th



promo

tion 

in al

l cla

sses’

 AP, 

indic

ating

 MAGI

CDRIV

E can

 inde

ed he

lp th

e tra

ining

 of p

ercep

tion



model

s.



我们使用表

中的 BE

VFusi

on 从 

nuSce

nes 验

证集中为 

3D 对象

检测提供每

类 AP7

。从结果中

，



我们观察到

，首先，对

于大型物体

的改进是显

著的，例如

公共汽车、

拖车和工程

车辆。其



次，外观不

太多样化的

物体，如交

通锥和障碍

物，表现出

更多的改善

，尤其是与

卡车相



比。第三，

我们注意到

这种改善对

汽车来说是

微不足道的

，而对行人

、摩托车和

自行车来



说是显著的

。这可能是

因为基线已

经表现良好

的汽车。对

于行人、摩

托车和自行

车，即使



距离自我汽

车较远的对

象生成的不

太真实，M

AGICD

RIVE 

也可以合成

自我汽车附

近的高质量



对象，如图

所示 17

-18。因

此，更准确

地检测自我

车附近的物

体有助于这

些类别的改

进。



总的来说，

地图的改进

伴随着所有

课程 AP

 的提升，

表明 MA

GICDR

IVE 确

实可以帮助

感知模



型的训练。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



× ×



× ×



Metho

ds re

solut

ion



mIOU 

for 6

 clas

ses



Metho

ds re

solut

ion



mIOU 

for 6

 clas

ses



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



Q MOR

E RES

ULTS 

FOR B

EV SE

GMENT

ATION



R BEV

 分割的更

多结果



BEVFu

sion 

is al

so ca

pable

 of B

EV se

gment

ation

 and 

consi

ders 

most 

of th

e cla

sses 

we us

ed in

 the



BEV m

ap co

nditi

on. D

ue to

 the 

lack 

of ba

selin

es, w

e pre

sent 

the r

esult

s in 

Table

 8 to

 faci

litat

e



com- 

paris

on fo

r fut

ure w

orks.

 As c

an be

 seen

, the

 272 

736 r

esolu

tion 

does 

not o

utper

form 

the



224 4

00 re

solut

ion. 

This 

is co

nsist

ent w

ith t

he re

sults

 from

 CVT 

in Ta

ble 1

 on t

he Ro

ad se

gment

.



Such 

resul

ts co

nfirm

 that

 bett

er ma

p con

trols

 rely

 on m

ainta

ining

 the 

origi

nal a

spect

 rati

o for



gener

ation

 trai

ning 

(i.e.

, avo

iding

 crop

ping 

on ea

ch si

de).



BEVFu

sion 

还能够进行

 BEV 

分割，并考

虑我们在 

BEV 映

射条件中使

用的大多数

类别。由于

缺



乏基线，我

们在表中给

出了结果 

8 以便于

将来工作的

比较。可以

看出，27

2 736

 的分辨率

并没



有优于 2

24 40

0 的分辨

率。这与表

中 CVT

 的结果一

致 1 在

路段上。这

样的结果证

实了更好的



地图控制依

赖于保持生

成训练的原

始纵横比(

即，避免在

每一侧裁剪

)。



Table

 8: G

enera

tion 

fidel

ity t

o BEV

 map 

condi

tions

. Res

ults 

are t

ested

 with

 BEVF

usion

 for 

BEV



segme

ntati

on on

 the 

nuSce

nes v

alida

tion 

set.



表 BEV

 图条件下

的发电保真

度。在 n

uScen

es 验证

集上使用 

BEVFu

sion 

对 BEV

 分割的结

果进



行测试。



CAM-o

nly C

AM+Li

DAR



仅限 CA

M 摄像头

+激光



雷达



Oracl

e - 5

7.09 

62.94



Oracl

e 224

×400 

52.72

 58.4

9



神谕 - 

57.09

 62.9



4



神谕 22

4×400

 52.7

2 58.

4



9



MAGIC

DRIVE



MAGIC

DRIVE



224×4

00



272×7

36



30.24

 48.2

1



28.71

 47.1

2



魔法驱动



魔法驱动



224×4

00



272×7

36



30.24

 48.2

1



28.71

 47.1

2



S GEN

ERALI

ZATIO

N OF 

CAMER

A PAR

AMETE

RS



T 相机参

数的泛化



To im

prove

 gene

raliz

ation

 abil

ity, 

MAGIC

DRIVE

 enco

des r

aw ca

mera 

intri

nsic 

and e

xtrin

sic p

arame

ters 

for d

iffer

ent p

erspe

ctive

s. Ho

wever

, the

 gene

raliz

ation

 abil

ity i

s som

ewhat

 limi

ted d

ue



to nu

Scene

s fix

ing c

amera

 pose

s for

 diff

erent

 scen

es. N

evert

heles

s, we

 atte

mpt t

o exc

hange

 the



intri

nsic 

and e

xtrin

sic p

arame

ters 

betwe

en th

e thr

ee fr

ont c

amera

s and

 thre

e bac

k cam

eras.

 The



compa

rison

 is s

hown 

in Fi

gure 

14. S

ince 

the p

ositi

ons o

f the

 nuSc

enes 

camer

as ar

e not



symme

trica

l



为了提高泛

化能力，M

AGICD

RIVE 

对不同视角

的原始相机

内部和外部

参数进行编

码。然而，



由于 nu

Scene

s 为不同

的场景固定

相机姿态，

泛化能力有

些有限。然

而，我们试

图在三个前



置摄像头和

三个后置摄

像头之间交

换内部和外

部参数。比

较如图所示

 14。因

为 nuS

cenes

 摄



像机的位置

是不对称的



from 

front

 to b

ack, 

and t

he ba

ck ca

mera 

has a

 120◦

 FOV 

compa

red t

o the

 70◦ 

FOV o

f the

 othe

r



cam- 

eras,

 clea

r dif

feren

ces b

etwee

n fro

nt an

d bac

k vie

ws ca

n be 

obser

ved f

or th

e sam

e 3D



coord

inate

s.



70° 1

20°



70° 1

20°



120° 

70°



120° 

70°



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



从前到后，

与其他摄像

机的 70

° FOV

 相比，后

摄像机具有

 120°

的 FOV

，对于相同

的 3D 

坐



标，可以观

察到前视图

和后视图之

间的明显差

异。



Figur

e 14:

 To s

how t

he ge

neral

izati

on ab

ility

 of l

earne

d cam

era e

ncodi

ng, w

e exc

hange

 the



camer

a par

amete

rs be

tween

 3 fr

ont c

amera

s and

 3 ba

ck ca

meras

. The

 3D p

ositi

on is

 the 

same 

for



two g

enera

tions

. We 

highl

ight 

some 

areas

 (wit

h box

 boxe

s)



图 14:

为了显示学

习的摄像机

编码的泛化

能力，我们

在 3 个

前置摄像机

和 3 个

后置摄像机

之



间交换摄像

机参数。两

代人的 3

D 位置是

相同的。我

们突出显示

一些区域(

用方框框)



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



U MOR

E GEN

ERATI

ON RE

SULTS



V 更多生

成结果



We sh

ow so

me co

rner-

case 

(Li e

t al.

, 202

2) ge

nerat

ions 

in Fi

gure 

15, a

nd mo

re ge

nerat

ions 

in



Figur

e 16-

Figur

e 18.



我们展示了

一些典型案

例(Li 

et al

.,202

2)几代人

在图 15

，以及图中

的更多代 

16-图 

18。



Figur

e 15:

 Gene

ratio

n fro

m MAG

ICDRI

VE wi

th co

rner-

case 

annot

ation

s.



图 15:

从 MAG

ICDRI

VE 生成

带有角案例

注释的代码

。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



Figur

e 16:

 Gene

ratio

n fro

m MAG

ICDRI

VE wi

th an

notat

ions 

from 

nuSce

nes v

alida

tion 

set.



图 16:

使用 nu

Scene

s 验证集

的注释从 

MAGIC

DRIVE

 生成。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



Figur

e 17:

 Gene

ratio

n fro

m MAG

ICDRI

VE wi

th an

notat

ions 

from 

nuSce

nes v

alida

tion 

set.



图 17:

使用 nu

Scene

s 验证集

的注释从 

MAGIC

DRIVE

 生成。



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



Publi

shed 

as a 

confe

rence

 pape

r at 

ICLR



4



Figur

e 18:

 Gene

ratio

n fro

m MAG

ICDRI

VE wi

th an

notat

ions 

from 

nuSce

nes v

alida

tion 

set.



图 18:

使用 nu

Scene

s 验证集

的注释从 

MAGIC

DRIVE

 生成。

