# M13.14-r3 development result

Date: 2026-08-19

Status: **development negative; confirmation remains sealed**.

The exact frozen r3 development stage completed all fits, evaluations, and
strict replays. The modular anchored candidate failed the all-replica hard gate
and did not consistently beat the monolithic anchored control.

A post-run verifier also found a non-promoting schema-normalization defect:
each report artifact summary includes its valid `schema_version`, while the
verifier's canonical reconstruction omitted that one field. All shared artifact
fields, file hashes, loaded policy fingerprints, and evaluation specs match.
This defect prevents the report from authorizing a later stage, but does not
change the already-failed development gate.

## Execution integrity

- teacher dataset: fresh r3 fit partition only
- real fit jobs: `12` (`4` seeds x `3` fitted arms)
- scored policy episodes: `1,280`
- shared diagnostics: `80` scripted ceiling + `80` mask-random
- strict replay count: `1,440/1,440`
- integrity violations: `0`
- replay violations: `0`
- unsafe-WAIT fraction: `0` in every candidate row
- scripted ceiling: `80/80`
- mask-random full objective: `0/80`

Protocol fingerprint:
`d4c309c0e9637727f555abc44064e206f6f11a29f58113da9884afd0cb3f38bb`.

Report content hash:
`40d9ba8a410913b84967cf1a8c9953db7713a199265a7be610c325a89b0843db`.

File and aggregate SHA-256 values:

- report file: `f0bfea8db5363be5034fe40e360438319756d9a1a83c0a44fa918375916ea74d`
- stage ledger: `1a6b49fe3c721ab3982c274ad54472c598a98c7add9313eec86bcaf224439f3c`
- teacher dataset: `f3a6d3d76de5e27122d044b2799dfebab31b9de14490e100a7e8a3c15f1a0e99`
- policy-artifact hash list: `15b0d83d91438c04f94e67c2b833d0000fc2b0d2fe56a6860d2038fcd8e3e954`
- trace hash list: `8734e1ac06a7b51de0dabb032325fb24cf98dc2e219fb97f10681d6103cc8439`

## Primary result

Full-objective successes out of 80 episodes per seed:

| Training seed | Modular + anchor + shield | Monolithic + anchor + shield | Modular, no persistent anchor | Modular + anchor, no shield |
| --- | ---: | ---: | ---: | ---: |
| `20261435` | `49/80` | `71/80` | `0/80` | `5/80` |
| `20261436` | `67/80` | `52/80` | `0/80` | `10/80` |
| `20261437` | `73/80` | `70/80` | `45/80` | `30/80` |
| `20261438` | `69/80` | `80/80` | `0/80` | `12/80` |
| **Total** | **`258/320`** | **`273/320`** | **`45/320`** | **`57/320`** |

The candidate therefore showed large aggregate advantages over its no-anchor
and no-shield controls, supporting the importance of persistent teacher
anchoring and the acting-loop shield. It did not support the modular
decomposition hypothesis: the monolithic anchored control scored higher in
aggregate and beat the candidate in two of four paired seeds.

## Candidate condition failures

Candidate full-objective successes by condition:

| Seed | Persistent reference | Renewal/morphology | Event relocation | Compound |
| --- | ---: | ---: | ---: | ---: |
| `20261435` | `17/20` | `12/20` | `19/20` | `1/20` |
| `20261436` | `20/20` | `17/20` | `10/20` | `20/20` |
| `20261437` | `15/20` | `18/20` | `20/20` | `20/20` |
| `20261438` | `20/20` | `20/20` | `9/20` | `20/20` |
| **Total** | **`72/80`** | **`67/80`** | **`58/80`** | **`61/80`** |

No candidate replica passed the required `18/20` threshold in every condition.
The failure mode varied materially by seed: compound collapse, event-relocation
collapse, or a smaller persistent/renewal miss. Safety and recovery accounting
were generally strong, so this remains a stability/scheduling failure rather
than a WAIT attractor or replay failure.

## Decision

- Do not open confirmation or audit.
- Do not select seed `20261437` or reinterpret aggregate `258/320` as a pass.
- Do not tune on r3 check traces or rerun the opened split.
- Preserve the result as evidence that persistent teacher anchoring and the
  shield are causally useful, while the declared modular decomposition is not a
  stable improvement over the monolithic anchored control.
- Any future learned-policy hypothesis requires a new protocol and fresh data;
  it should use the monolithic anchored control as the stronger starting point.

