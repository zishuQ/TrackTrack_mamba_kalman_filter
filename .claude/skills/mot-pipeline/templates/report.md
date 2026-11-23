# {{dataset}} experiment: {{experiment}}

- base: {{base_experiment}} | status: {{status}} | reuse_from: {{reuse_from}}
- config: `{{config}}`
- reason: {{change_reason}}
- checkpoint: `{{checkpoint}}` (sha256 {{checkpoint_sha256}})
- zip: `{{zip}}` (sha256 {{zip_sha256}})

## Official scores

| metric | value | vs best | vs base |
|---|---:|---:|---:|
| HOTA | {{HOTA}} | {{dHOTA_best}} | {{dHOTA_base}} |
| AssA | {{AssA}} | - | {{dAssA_base}} |
| DetA | {{DetA}} | - | {{dDetA_base}} |
| IDF1 | {{IDF1}} | - | {{dIDF1_base}} |
| MOTA | {{MOTA}} | - | {{dMOTA_base}} |
| IDSW | {{IDSW}} | - | {{dIDSW_base}} |
| Frag | {{Frag}} | - | {{dFrag_base}} |

## Gates

- motion min/mean/max: {{motion_min}}/{{motion_mean}}/{{motion_max}}
- appearance min/mean/max: {{appearance_min}}/{{appearance_mean}}/{{appearance_max}}

## Commands and stages

{{stage_list}}

## Suggestions

{{suggestions}}
