'''
code taken from the following and modified.

[1] https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

[2] https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/uper_head.py

Other references:

[1] https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/psp_head.py


Copyright 2020 The MMSegmentation Authors. All rights reserved.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2020 The MMSegmentation Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.



MIT License

Copyright (c) 2020 Yassine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

   '''


#from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation
#from transformers import AutoImageProcessor, SwinModel
from torchvision import models
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer import SwinTransformer
from load_pretrained import load_pretrained, weights_init
import numpy as np

#backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

#config = UperNetConfig(backbone_config=backbone_config)
#model = UperNetForSemanticSegmentation(config)
#model2 = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")
#print('type(model): ', type(model))
#print('model: ', model)
#print('model2: ', model2)

#model3 = getattr(models, 'resnet101')(True)
#model4 = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

#print('type(model4): ', type(model4))
#print('model4: ', model4)
#print('model3 children: ', list(model3.children()))
#print('model3 children: ', model3.children())
#print('model4.config: ', model4.config)
#summary(model4, input_size = (3,224,224), batch_size=4)

class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3, dilation=1, output=64, layertype=1, droupout=False):
        super(ConvLayer, self).__init__()
        if droupout == False:
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        else: 
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))

        self.layer4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer5 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        #print('ConvLayer: ')
        #print('x shape: ', x.shape)
        out1 = self.layer1(x)
        #print('out1 shape: ', out1.shape)
        if self.layertype == 1:
            out1 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out1, inds = self.layer4(out1)
            #print('out3 shape: ', out3.shape)
            return out1, inds
        elif self.layertype == 2:
            out1 = self.layer2(out1)
            #print('out2 shape: ', out2.shape)
            out1 = self.layer3(out1)
            #print('out3 shape: ', out3.shape)
            out1, inds = self.layer4(out1)
            #print('out4 shape: ', out4.shape)
            return out1, inds
        elif self.layertype == 3:
            out1 = self.layer3(out1)
            return out1
        elif self.layertype == 4:
            out1 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out1 = self.layer5(out1)
            #print('out3 shape: ', out3.shape)
            return out1


class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassifyBlock, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.layerprob = nn.Softmax(dim=1)
        '''
        torch.nn.init.normal_(self.layer.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layerprob.weight, mean=0, std=1)
        '''

    def forward(self, x):
        #print('ClassifyBlock: ')
        #print('x shape: ', x.shape)
        out = self.layer(x)   
        #print('out shape: ', out.shape)
        #print('breakpoint 1:' )
        #breakpoint()
        #out = torch.permute(out, (0,3,1,2))
        #print('out shape: ', out.shape)
        out = self.layerprob(out)
   #     #print('out shape: ', out.shape)
        #print('out[0,:,0,10]: ', out[0,:,0:2,10])
        #print('torch.sum(out[0,:,0,10]): ', torch.sum(out[0,:,0,10]))
        return out

class PSPhead(nn.Module):
    def __init__(self, input_dim=1024, output_dims=256, final_output_dims=1024, pool_scales=[1,2,3,6]):
        super(PSPhead, self).__init__()
        self.ppm_modules = [nn.Sequential(nn.AdaptiveAvgPool2d(pool), nn.Conv2d(input_dim, output_dims, kernel_size=1),
            nn.BatchNorm2d(output_dims),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) for pool in pool_scales]

        self.bottleneck = nn.Sequential(nn.Conv2d(input_dim + output_dims*len(pool_scales), final_output_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_output_dims),
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))


    def forward(self, x):
        x = x.permute((0,3,1,2))
        ppm_outs = []
        ppm_outs.append(x)
        for ppm in self.ppm_modules:
            #ppm_out = Resize((x.shape[2], x.shape[3]), interpolation=InterpolationMode.BILINEAR)
            #print('ppm(x).shape: ', ppm(x).shape)
            ppm_out = F.interpolate(ppm(x), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=None)
            ppm_outs.append(ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        #print('ppm_outs shape: ', ppm_outs.shape)
        ppm_head_out = self.bottleneck(ppm_outs)
        #ppm_head_out = ppm_head_out.permute((0,2,3,1))
        #print('ppm_head_out shape: ', ppm_head_out.shape)
        return ppm_head_out

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        #P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        #print('len(features): ', len(features))
        P = []
        for i in reversed(range(1, len(features))):
            features[i-1] = F.interpolate(features[i], size=(features[i-1].shape[2], features[i-1].shape[3]), mode='bilinear', align_corners=True) + features[i-1] 
            P.append(features[i-1])
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        #print('len(P): ', len(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        #print('P[0].shape: ', P[0].shape)
        H, W = P[0].shape[2], P[0].shape[3]
        #print('H: ', H, ' W: ', W)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]
        #print('len(P): ', len(P))
        x = self.conv_fusion(torch.cat((P), dim=1))
        #print('x.shape: ', x.shape)
        return x

class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes, in_channels=3, backbone='swinv1_7_224', pretrained=True, use_aux=True, fpn_out=256, freeze_bn=False, head_out=128, backbonepath='swin_tiny_patch4_window7_224.pth', **_):
        super(UperNet, self).__init__()

        if backbone == 'resnet34' or backbone == 'resnet18':
            feature_channels = [64, 128, 256, 512]
            self.backbone = ResNet(in_channels, pretrained=pretrained)
        elif backbone == 'swinv1_7_224':
            feature_channels = [96, 192, 384, 768]
            self.backbone = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=16,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0,
                                drop_path_rate=0.1,
                                ape=False,
                                #norm_layer=layernorm,
                                patch_norm=True,
                                use_checkpoint=False,
                                fused_window_process=False)
            if pretrained:
                load_pretrained(backbonepath, self.backbone, classification=False)

        else:
            feature_channels = [256, 512, 1024, 2048]

        #self.PPN = PSPModule(feature_channels[-1])
        self.PPMhead = PSPhead(input_dim=768, output_dims=96, final_output_dims=768)
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = ConvLayer(fpn_out, head_out, kernel_size=3, stride=1, padding=1,  output=64, layertype=3, droupout=True)
        self.ClassifyBlock = ClassifyBlock(64, num_classes)
        #self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        #if freeze_bn: self.freeze_bn()
        #if freeze_backbone: 
        #    set_trainable([self.backbone], False)
        print('applying weights_init to PPMhead')
        self.PPMhead.apply(weights_init)
        print('applying weights_init to FPN')
        self.FPN.apply(weights_init)
        print('applying weights_init to head')
        self.head.apply(weights_init)
        print('applying weights_init to ClassifyBlock')
        self.ClassifyBlock.apply(weights_init)
        self.num_classes = num_classes


    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        for i in range(len(features)):
            h = int(np.sqrt(features[i].shape[1]))
            features[i] = features[i].view(features[i].shape[0], h, h, features[i].shape[2])
            if i != len(features) - 1:
               features[i] = features[i].permute(0,3,1,2)
            #print('after backbone, features i shape: ', features[i].shape)
        #print('features[-1] shape before PPMhead: ', features[-1].shape)
        features[-1] = self.PPMhead(features[-1])
        #print('features[-1] shape after PPMhead: ', features[-1].shape)
        x = self.FPN(features)
        #print('after FPN x.shape: ', x.shape)
        x = self.head(x)
        #print('after head ConvLayer x.shape: ', x.shape)
        x = F.interpolate(x, size=input_size, mode='bilinear')
        #print('after interpolate x.shape: ', x.shape)
        x = self.ClassifyBlock(x)
        #print('after ClassifyBlock x.shape: ', x.shape)
        return x
