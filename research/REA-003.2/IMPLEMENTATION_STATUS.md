# Implementação

| Frente | Estado |
| --- | --- |
| Preservação | Worktrees A e B independentes. O HEAD e os binários anteriores continuam na árvore original. Cache histórico copiado fora do Git. |
| Diagnóstico P0 | Logs frios, parcialmente aquecidos e históricos auditados. Hipóteses separadas de observações em runtime. |
| Upstream B | Vice-Tay compilado no SHA auditado. Referência independente. |
| Integração P0 | Cache staging, testes de identidade por formato, serialização e retries, política de histórico, guard de compilação e diagnóstico PSV0. |
| Integração P1 e compatibilidade semântica | Candidatos auditados. Coerência de heap, journal, aliases, readbacks e mudanças Wave32 dependem de gate e testes adicionais. |
| Consolidação | Release gerada e sete ensaios de inicialização documentados. Testes de CPU, GPU, integração e Clippy executados. Gate 1 permanece aberto. |

Os caminhos DXR, upscale, detiling, GPU authority, sparse, readback, contabilidade, render graph, shader recovery, GPU-driven e fast paths anteriores continuam no histórico e na árvore fonte. Os lançadores da candidata ativam o mínimo de opções REA necessárias ao experimento P0.
