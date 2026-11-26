# Diagnóstico das execuções locais da REA-003.1

A regressão aparece no executável congelado e no executável da REA-003.1. O mecanismo imediato mais bem sustentado é a espera por PSOs durante a transição para cenas mais pesadas. A criação de PSOs no driver foi a etapa de compilação com maior tempo acumulado nas quatro execuções que chegaram à carga pesada. Cache frio ou parcial contribuiu para a recompilação. As sessões disponíveis não isolam o efeito do cache, da configuração, do estado persistente e da pressão de RAM.

## Cobertura e método

Foram processados até o fim 87 arquivos originais de `%LOCALAPPDATA%/victory/logs`, incluindo os 15 arquivos das cinco execuções pedidas, os três logs de tradução e 69 arquivos históricos de 26 execuções REA-002 e REA-003. O [BENCHMARK_RESULTS.json](BENCHMARK_RESULTS.json) registra contagem de linhas, SHA-256, sidecars, janelas estruturadas da REA-003.1, série de compilação e contexto histórico. Dados de fontes em UTF-16 foram decodificados pela BOM. A agregação preserva o valor ausente sem preenchê-lo com zero e conserva a distinção entre as últimas linhas de telemetria, de salvamento e do processo. Os horários dos logs estão em UTC.

## Perfil e progresso registrados

As cinco execuções usaram perfil Series S, render scale solicitado de 0,4 e upscaler do próprio título. Os sidecars preservam as diferenças de flags, diretórios de cache e executáveis. O binário da baseline mantém o SHA-256 `1387ec6e3c49acc53697c9bf9a0bccd3f3e83e7dc46a7fdb379472953bbd3735`, que coincide com o manifesto da REA-003.1 para `f11885b086aa002662da5591c31029accb3b2e98`. O sidecar registrou `2b27ce3688e7` como checkout do launcher. Stable e Experimental registraram esse commit com binário de SHA-256 `665b79f8625bd7de4805bacee5adf9ed4024472fcab810d1ff47dae4ad57c05c`.

| Execução | FPS na última janela | ms/frame na última janela | Janelas com ≥1.000 draws/frame | Biblioteca aberta | Pico de pendências | Pendências finais |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Baseline 191115 | 0,081 | 12.391,038 | 6 de 21 | Vazia, 0 chaves antecipadas | 1.129 | 736 |
| Stable 191520 | 0,082 | 12.259,558 | 2 de 15 | Vazia, 0 chaves antecipadas | 1.122 | 1.113 |
| Experimental 191733 | 29,981 | 33,354 | 0 de 7 | Vazia, 0 chaves antecipadas | 0 | 0 |
| Experimental 191836 | 0,041 | 24.653,552 | 1 de 14 | 5.870 B, 1 chave antecipada | 729 | 729 |
| Stable 192558 | 0,074 | 13.556,530 | 12 de 33 | 13.682.300 B, 1.438 chaves antecipadas | 1.113 | 309 |

A Experimental 191733 terminou com aproximadamente 11 draws/frame e `^C`, antes de alcançar uma janela de carga comparável. Baseline 191115 e Stable 191520 também terminam com `^C`. Experimental 191836 e Stable 192558 terminam com linhas de workers ainda registrando traduções ou PSOs, sem confirmação de saída ordenada. A busca integral não encontrou a assinatura de remoção ou hang do dispositivo nessas cinco execuções. As janelas de carga são aproximações pela contagem de draws. O log não estabelece em que instante apareceram cutscene, mundo ou prisão, nem se a imagem era íntegra. Os campos de quantis da amostra de pacing e os FPS das janelas têm janelas e denominadores distintos.

## Evolução da compilação

| Execução | PSOs criados e tempo do driver | Traduções e tempo | Espera por PSO | Solicitações descartadas por pendência | Salvamento final |
| --- | --- | --- | ---: | ---: | --- |
| Baseline 191115 | 754 / 501.232,4 ms | 980 / 81.697,1 ms | 726.701,5 ms | 27.412 | 64.756.868 B, 755 criados, 50 carregados |
| Stable 191520 | 302 / 96.818,7 ms | 415 / 7.953,0 ms | 134.958,0 ms | 12.368 | 13.682.300 B, 310 criados, 19 carregados |
| Experimental 191733 | 4 / 6,6 ms | 7 / 13,0 ms | 35,6 ms | 0 | 5.870 B, 1 criado, 0 carregados |
| Experimental 191836 | 719 / 200.386,5 ms | 948 / 35.023,6 ms | 287.339,8 ms | 8.696 | 46.768.054 B, 725 criados, 51 carregados |
| Stable 192558 | 925 / 1.192.079,7 ms | 852 / 153.732,0 ms | 973.370,0 ms | 26.217 | 129.933.708 B, 940 criados, 291 carregados |

Os acumulados acima vêm da última linha de telemetria de cada execução. O salvamento pode ocorrer antes ou depois dessa linha. `driver_compile_total_ms` soma tempos de chamadas a criação no driver, inclusive tentativas com erro, e `driver_compiled` conta sucessos. `queued` inclui tanto jobs à espera quanto jobs em execução. `pending_drops` conta usos que voltaram sem PSO pronto em `ready`, de modo que o mesmo PSO pode aparecer várias vezes. Drops de draws e dispatches no JSONL têm outra instrumentação e não servem como substituto numérico para esse contador. Os tempos do driver, da tradução e das esperas podem se sobrepor entre threads.

Na Stable 192558, a biblioteca parcial foi aberta às 22:26:05 UTC. Às 22:26:10, havia 1.037 pendências, 26 recargas e 36 criações novas, mesmo com 1.100 PSOs já iniciados antecipadamente. O pico de 1.113 pendências veio às 22:26:41 e antecedeu os primeiros drops por PSO às 22:28:19. A linha 6.181 às 22:31:39, citada no pedido, registrou 830 criações, 740 traduções, 25.255 drops e 432 pendências. Depois dela, a linha 6.484 às 22:31:52 registrou mais 95 criações, 112 traduções e 962 drops. A biblioteca final de 129.933.708 B foi salva às 22:32:08 com 1.617 chaves e 1.352 inícios antecipados.

No fim da Stable 192558, o tempo acumulado de criação de PSO foi cerca de 7,75 vezes o de tradução. Na janela final, o temporizador de GPU mediu 89,214 ms por frame frente a 13.556,530 ms de parede. A amostra indica pouca execução gráfica registrada em relação à latência do frame. O contador não mede ocupação de todos os motores da GPU. Os logs registram orçamento-base de 12.000 ms por frame e por tipo de pipeline. Isso é consistente com vários frames próximos de 12 s e outros perto de 24 s quando há esperas de tipos diferentes, embora não feche toda a contabilidade de latência. Reduzir esse orçamento por si só pode trocar espera por mais descartes de trabalho necessário à cena.

## Caches históricos e variáveis de comparação

A REA-002 Conservative de 10:27:09 abriu uma biblioteca de 500.759.564 B e 2.848 chaves antecipadas. Terminou em 4,92 FPS e teve 19 janelas de pelo menos 1.000 draws/frame com mediana de 3,52 FPS. Seu sidecar também indica Series S, scale 0,4, upscaler próprio e as seis flags REA da execução Baseline 191115. A T10 de 10:23:10 abriu 500.545.812 B e 2.833 chaves e registrou mediana de 4,51 FPS em 21 janelas de carga. A REA-003 Baseline de 15:04:02 abriu 510.409.866 B e 2.851 chaves e terminou em 4,16 FPS. A REA-003 Integrity do mesmo dia abriu biblioteca de 510.409.866 B e terminou em 2,09 FPS. Esses sidecars históricos não fixam o hash da build nem identificam a mesma cena em todos os casos.

Na série de 19 h, Baseline 191115 e Stable 191520 abriram bibliotecas vazias, com zero acertos de DXIL persistente. O salvamento final da Stable 191520 coincide em tamanho com a biblioteca aberta pela Stable 192558 no mesmo bucket e com o mesmo executável, com 1.438 chaves antecipadas preservadas. A segunda execução registrou 540 acertos de shader, 291 cargas bem-sucedidas de PSO e 925 criações ainda necessárias. A Experimental 191836 reutilizou a biblioteca de 5.870 B deixada pela sessão curta. A biblioteca carregada confirma reaproveitamento parcial e os contadores de cargas incluem entradas criadas e lidas durante o próprio processo. A coleção de sessões sustenta a influência do cache frio, mas falta o A/B do mesmo binário, com mesma cena e estado, entre frio e cópia comprovadamente compatível da biblioteca histórica.

O código existente mantém jobs solicitados à frente dos especulativos ainda enfileirados. Um job que já entrou no driver não é preemptível por esse scheduler. A carga antecipada volumosa e os PSOs caros podem ocupar todos os workers disponíveis antes do primeiro uso crítico. Os dados sustentam testar limite de admissão especulativa e capacidade reservada para jobs do frame, medindo progresso, latência e drops em vez de encurtar o timeout.

## Tradução offline

O corpus Safe registrou 26.177 programas traduzidos de 26.177 considerados na etapa de tradução e 25.638 PSOs criados, incluindo 2.700 compute. No `XboxBvhBuilder.dll`, 52 de 52 programas foram traduzidos e validados em ambos os perfis. O Baseline criou 43 PSOs e recusou nove no driver. Safe criou 52 e registrou zero dessas recusas. Esses testes evidenciam compatibilidade de tradução e criação offline, e não medem submissões, sincronização de recursos, correção visual ou progressão em gameplay.


## Fase anterior à compilação nos ensaios REA-003.2

Na montagem WinFsp somente leitura, o primeiro marcador de apresentação fria foi registrado após 86,394 s na baseline congelada, 73,130 s na candidata e 13,683 s no Vice-Tay. Nesse ponto o contador de draws era zero. Os três primeiros marcadores precedem a primeira abertura da respectiva biblioteca de PSOs. Tempos para primeiro Present não representam tempos para cutscene ou prisão.

Entre a criação das filas DirectStorage e a primeira atividade de pipeline, houve 68,45 s na candidata e aproximadamente 5,8 s no Vice-Tay. Na baseline congelada passaram 77,75 s desde a ativação do modo assíncrono de DirectStorage até o primeiro warning de pipeline. A candidata consumiu aproximadamente 72 segundos de CPU em 72 segundos de parede nessa fase, com pico amostrado de 12,99% dos oito processadores lógicos e pelo menos 4.641 MiB físicos disponíveis. Uma amostra posterior identificou uma thread com 70,64 s cumulativos de CPU em um total de 87,28 s do processo.

O Vice-Tay inicia sete threads auxiliares para mapeamento, trabalha com grupos de páginas XMem e utiliza oito workers DirectStorage, frente a quatro nos executáveis locais. Os caminhos diferem também em agrupamento de chamadas de mapeamento, resolução de páginas, handles de I/O por worker e ordenação de submit. Os commits 565f4a56, 49d71266, fbc18308 e 68e10e57 permitem separar esses deltas. A execução não coletou pilha da thread quente. A concentração em uma thread contextualiza o atraso atual, enquanto os logs históricos de transição 3D medem outro estágio com recompilação pesada.

O A/B curto no mesmo Vice-Tay, com WinFsp controlado pelo lançador e cache vazio, registrou 14,887 s com o padrão de oito threads de mapeamento e 16,013 s com uma. O processo com uma thread manteve 2,41 GiB físicos disponíveis no mínimo amostrado, frente a 0,40 GiB no padrão. Houve uma amostra de cada configuração. O tempo observado junto da configuração de uma thread continua bem abaixo dos 69–73 s locais. Os outros deltas de XMem e DirectStorage precisam de isolamento para atribuição causal.
