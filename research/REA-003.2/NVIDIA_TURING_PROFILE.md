# Perfil NVIDIA Turing de 8 GiB

Host medido com RTX 2060 SUPER, 8.192 MiB e driver 617.42. A baseline histórica tem Series S e escala 0,4. A Stable usa Series S, escala 0,4, dois workers de compilação e guard de memória e fila. A Experimental repete o mesmo binário e política e desabilita somente a leitura e gravação de histórico antecipado. O cache binário de driver continua disponível. Os três perfis registram binário, SHA, variáveis e seeds. Os stubs de iluminação usados na baseline histórica não entram na candidata. Essa diferença visual impede interpretar Baseline e Stable como A/B puro de desempenho.

O perfil inicia com parâmetros conservadores de buffers e staging do preset histórico, sem aumentar o orçamento pelo tamanho nominal da GPU. Não altera o pagefile, o TDR ou o cache oficial NVIDIA. No log da Stable 192558, Wave32 é o máximo do dispositivo. A compatibilidade de cada PSO com Wave64 deve ser avaliada por shader e por contrato, usando o diagnóstico incluído.

Para uma validação manual, rode Stable até a primeira cena 3D e a prisão e encerre normalmente para salvar a biblioteca. Use uma nova execução com o cache da sessão anterior como seed. Use Experimental com a mesma seed inicial somente se a fila de PSOs especulativos continuar dominante. Compare a mesma cena e as mesmas opções.

Use `-GameDir` para uma cópia local íntegra ou `-ReadOnlySource` para expor a origem local por WinFsp/rclone sem gravar nela. Os três perfis aceitam `-BudgetSeconds`, `-CacheSeed` e `-DataSeed`. A montagem temporária introduz uma camada de I/O, de modo que os tempos absolutos devem ser comparados com outra execução pela mesma montagem.

```powershell
$title = 'C:\Users\gabeg\Downloads\GTA6.Xbox.v1.0.2.1'
$seed = Join-Path $env:LOCALAPPDATA 'victory\rea0032\seeds\historical-default'
& .\scripts\run-rea0032-stable.ps1 -ReadOnlySource $title -DataSeed (Join-Path $seed 'data') -Verbose
```
