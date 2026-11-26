# Pressão de RAM e concorrência

A Stable 192558 chegou a 3.160 MiB de shadow RAM contabilizada, 4.093 MiB de uso DXGI não local e 4.441 MiB de uso local, com orçamento local de 7.238 MiB. Essas séries medem objetos e segmentos diferentes e podem compartilhar backing físico. Os logs da execução não trazem os dados de memória do processo e do sistema necessários para calcular a pressão total, nem registram faults com I/O para confirmar paginação.

## Picos amostrados

| Execução | DXGI local (MiB) | DXGI não local (MiB) | Shadow RAM (MiB) | Retidos aguardando fence, objetos |
| --- | ---: | ---: | ---: | ---: |
| Baseline 191115 | 6.153 | 3.987 (log textual) | 3.213 | Campo ausente |
| Stable 191520 | 4.541 | 3.926 | 3.113 | 294 |
| Experimental 191733 | 750 | 662 | 186 | 0 |
| Experimental 191836 | 4.667 | 3.788 | 2.921 | 79 |
| Stable 192558 | 4.441 | 4.093 | 3.160 | 4.589 |

Os valores são máximos de cada série e podem vir de amostras diferentes. A primeira janela da Stable 192558 registrou 174 MiB de shadow RAM, 647 MiB não locais e 558 MiB locais. A última registrou 3.132, 4.086 e 4.429 MiB, respectivamente. O contador de aposentados passou de 101 para 4.589 entre as duas últimas amostras, afastadas por 13,557 s. Em paralelo, as pendências de compilação caíram de 432 para 309 e mais 95 PSOs foram criados. Trata-se de um ponto de acúmulo observado; a série não atribui bytes a cada objeto nem demonstra retenção permanente. A última linha do processo vem depois da última amostra e não registra a drenagem.

## Contadores e sobreposição

| Campo | Definição verificada no código |
| --- | --- |
| `shadow_ram_mb` | Soma incremental de `Shadow.len` dos shadows vivos registrados com backing CPU. A implementação exclui a parcela ReBAR e pode manter uma cópia em VRAM para os mesmos dados. É tamanho contabilizado de objetos, não observação de páginas físicas residentes. |
| `shadow_vram_mb` | Soma de `card.size` e da parcela ReBAR dos shadows registrados. Na Stable 192558, pico de 1.173 MiB. Pode integrar a categoria de uso local do DXGI. |
| `vram_used_mb` / `dxgi_nonlocal_used_mb` | `CurrentUsage` retornado pelo DXGI para o processo no adaptador, nos segmentos local e não local. O não local pode incluir heaps de upload e readback e outros recursos com backing de sistema. A origem do aumento não pode ser atribuída a spill apenas com esse campo. |
| `acc_comm_mb` / `acc_res_mb` | Bytes de texturas committed e reserved nos rastreadores internos. Reserved descreve extensão virtual de recursos. Sparse mapped reporta os tiles mapeados e o DXGI já pode contabilizar parte da memória física correspondente. |
| `staging_mb` | Campo do rastreador de staging na janela. Pico de 79 MiB na Stable 192558. A alocação pode pertencer a heaps já observados pelo DXGI e representa apenas parte dos caminhos de upload. |
| `upload_ring_mb` | Indicador baseado em `ring_bytes_suballoc` da arena de upload. Foi zero na série Stable 192558. Não cobre os demais caminhos de upload. |
| `retired_pending` | Número de objetos na fila de `host::retire` associados ao maior número de submissão em voo. `bury` decrementa ao retirar os concluídos, antes da destruição pelo worker `victory-retire`. O contador não contém peso em bytes nem mede toda a memória dos objetos já entregues à thread de descarte. |
| Shaders vivos | A tabela `SHADERS` segura contêineres e traduções em `Arc<Shader>` e `OnceLock<Option<Translated>>` durante o processo. A instrumentação diagnóstica distingue bytes dos contêineres, DXIL e biblioteca mapeada. Os logs release desta amostra não incluem essa linha. |
| IR temporário e scratch do driver | Durante a tradução e a chamada de criação de PSO há alocações temporárias fora dos totais de tamanho de DXIL. Nenhuma série da amostra as mede individualmente. O tempo de criação mede latência, não tamanho. |

As referências de implementação são [shadow.rs](../../crates/victory-gpu/src/shadow.rs), [host.rs](../../crates/victory-gpu/src/host.rs), [accounting.rs](../../crates/victory-gpu/src/accounting.rs), [rea_telemetry.rs](../../crates/victory-gpu/src/rea_telemetry.rs), [pipeline.rs](../../crates/victory-gpu/src/pipeline.rs) e [pipeline/library.rs](../../crates/victory-gpu/src/pipeline/library.rs). Os campos de tamanho terminados em `_mb` no JSONL usam divisores binários ou deslocamentos de 20 bits e estão apresentados aqui em MiB.

## CPU, workers e limites de evidência

- As cinco execuções registraram plano de quatro workers de shader, quatro de streaming, dois de shadow e um de memcopy. O log também confirma quatro threads DStorage iniciadas, que são os quatro workers de streaming desse plano. O código inicia a quantidade configurada de workers de PSO quando necessário. Tradução síncrona dentro do job e criação no driver podem ocupar o mesmo worker em sequência. As threads internas que o driver pode criar não foram amostradas.
- Há threads do título e, conforme o caminho em uso, relay, espera tardia, descarte, temporização de GPU e refreshes paralelos. A geração de trabalho de copy-back pode usar o pool shadow. O número planejado ou criado não mede quantas threads estavam executáveis numa janela. A configuração comporta mais trabalho simultâneo que os quatro núcleos físicos, e o custo da contenção precisa de amostras por thread.
- `cpu_prep_ms` agrega tempo de preparação de submissões por frame. Em diagnóstico, `copystat` calcula a diferença de tempo de CPU do processo em modo usuário e kernel via `GetProcessTimes` e divide pelo número de frames. Isso é trabalho somado de todas as threads, em ms/frame. Para expressar ocupação relativa a 100% da máquina, divide-se a diferença desse tempo pelo tempo de parede da mesma janela vezes oito processadores lógicos. Usar apenas tempo de parede resulta em unidades de um processador lógico, podendo ultrapassar 100%. O percentual reportado na observação externa não aparece com sua ferramenta e fórmula nos logs, portanto não foi renormalizado.
- A linha diagnóstica de memória usa `K32GetProcessMemoryInfo` para `WorkingSetSize` e `PagefileUsage` e `GlobalMemoryStatusEx` para RAM física disponível e capacidade de commit adicional disponível ao processo. Não apareceu nas cinco execuções release. `PageFaultCount`, quando disponível no diagnóstico, combina faults de naturezas distintas e não identifica hard faults ou I/O do pagefile. Também não há ETW ou contadores de sistema nessas sessões. O relato de RAM próxima da saturação permanece como observação do usuário.

## Hipótese testável

O cache vazio eliminou a reutilização inicial de PSOs e DXIL. O reaproveitamento parcial elevou o número de criações antecipadas concorrentes, com mais de mil jobs acumulados no início da Stable 192558. Isso sustenta a hipótese de concorrência e retenção durante a transição. O tamanho dos caches e os picos dos contadores fornecem motivos para testar essa hipótese, mas não comprovam que houve paginação intensa ou que compile workers foram o maior consumidor de RAM.

Para o A/B da build candidata, o suficiente é amostrar em intervalo fixo, inclusive quando o título deixa de apresentar frames. As chaves são relógio monotônico, PID e identidade da execução. Colete `PrivateUsage`, working set, RAM física livre, CommitTotal e CommitLimit, faults e I/O de armazenamento classificados por página e processo, contagem/tempo por tipo de job, idade da pendência e delta de fence. Correlacione a etapa de compilação com o pico de NonLocal, staging, shadow e descartes. Varie primeiro a concorrência especulativa sob as mesmas condições de cache, sem mudar pagefile ou knobs de residência na mesma comparação. Um limite adaptativo deve bloquear novos jobs especulativos antes de saturar a folga física ou de commit, preservando capacidade de progresso para PSOs solicitados e retomando com histerese.



## Amostras novas de inicialização

As amostras de CPU do lançador dividem a variação de CPU do processo pelos milissegundos da janela e pelos oito processadores lógicos. O processo da candidata ficou perto do equivalente a uma thread ocupada por completo antes do primeiro Present. A RAM física disponível ficou acima de 4,5 GiB nessa fase. O Vice-Tay atingiu um mínimo de aproximadamente 91 MiB livres em uma fase posterior do seu ensaio de 95 s, após avançar mais frames. Comparamos estágios distintos. Os números não atribuem pressão a um subsistema sozinho e não medem I/O de pagefile ou hard faults por processo.
