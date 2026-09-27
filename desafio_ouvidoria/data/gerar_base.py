"""Gera data/manifestacoes.json e data/duplicatas_gabarito.json.

O enunciado cita um arquivo manifestacoes.json fornecido pelo professor, mas ele
não veio junto com o material. Esta base sintética segue a especificação:
40 manifestações (M001–M040), 5 categorias oficiais, textos de 50 a 800
caracteres, ~15% de duplicatas semânticas (6 pares) e 5 textos longos (>500).
Para usar a base oficial, basta substituir manifestacoes.json (e atualizar o
gabarito de duplicatas).
"""
import json
from pathlib import Path

M = [
    ("2026-03-02", "infraestrutura", "Calçada toda quebrada na Rua das Flores, perto do número 120. Dois idosos já caíram tentando passar por ali e ninguém da prefeitura veio consertar."),
    ("2026-03-02", "saúde", "Fiquei mais de cinco horas esperando atendimento na UPA do Centro com meu filho de dois anos com febre alta. Só havia um clínico de plantão."),
    ("2026-03-03", "infraestrutura", "Tem um buraco enorme na Av. Brasil, em frente ao supermercado Bom Preço. Vários carros já estouraram o pneu e um motociclista caiu ontem."),
    ("2026-03-03", "segurança", "Os moradores da Rua Sete estão com medo porque aconteceram três assaltos nesta semana na esquina da padaria, sempre depois das dez da noite."),
    ("2026-03-04", "infraestrutura", "A Rua Pernambuco está completamente escura à noite porque os postes de iluminação estão apagados há mais de um mês."),
    ("2026-03-04", "educação", "A Escola Municipal Maria José está sem professor de matemática desde o início do semestre e os alunos do 8º ano estão sem aula dessa matéria."),
    ("2026-03-05", "meio ambiente", "Uma fábrica do distrito industrial solta fumaça preta todas as noites. O cheiro forte chega até as casas do bairro Vila Nova e as crianças acordam tossindo."),
    ("2026-03-05", "saúde", "O posto de saúde do Jardim América está sem médico há duas semanas. As pessoas vão até lá e voltam para casa sem consulta."),
    ("2026-03-06", "segurança", "Os carros passam em alta velocidade na frente da creche da Rua Ceará e não existe faixa de pedestre nem lombada. É questão de tempo até atropelarem uma criança."),
    ("2026-03-06", "saúde", "Sou diabético e dependo da insulina que a farmácia básica do posto do bairro Esperança deveria fornecer todo mês. Desde janeiro a farmácia diz que o medicamento está em falta e que não há previsão de chegada. Já fui lá quatro vezes, perdi dias de trabalho e em uma das visitas a funcionária disse para eu procurar outra unidade, que também estava sem estoque. Comprar na farmácia particular custa quase metade do meu salário. Minha glicemia está descontrolada e meu médico disse que posso ter complicações sérias. Peço que a Secretaria de Saúde regularize o abastecimento e informe à população quando os remédios chegam."),
    ("2026-03-09", "educação", "Os alunos da Escola Paulo Freire estão sem merenda escolar há uma semana. Muitas crianças só fazem uma refeição por dia, que é a da escola."),
    ("2026-03-09", "meio ambiente", "Estão cortando as árvores antigas da praça central sem nenhuma placa de autorização da prefeitura. Já derrubaram quatro ipês."),
    ("2026-03-10", "infraestrutura", "O bueiro da Rua Bahia está entupido e, quando chove, a água da rua invade as casas. Já perdemos móveis duas vezes neste ano."),
    ("2026-03-10", "meio ambiente", "Há lixo acumulado no terreno baldio da Rua Goiás há meses. O mato está alto, tem rato e o mau cheiro está insuportável."),
    ("2026-03-11", "segurança", "Um grupo usa drogas todas as noites na quadra de esportes do bairro Novo Horizonte e intimida quem passa, inclusive crianças que voltam da escola."),
    ("2026-03-11", "infraestrutura", "Moro na comunidade do Sítio Várzea e a única ligação com a cidade é uma ponte de madeira sobre o riacho que está com várias tábuas podres e sem proteção lateral. Quando chove, a estrada de terra que chega até a ponte vira lama e o ônibus escolar e o carro do lixo simplesmente param de passar. No mês passado uma moto caiu da ponte e o rapaz quebrou a perna. Os moradores já fizeram mutirão para trocar algumas tábuas, mas o problema é estrutural. Pedimos a construção de uma ponte de concreto e o cascalhamento da estrada antes do próximo inverno, porque a comunidade fica isolada."),
    ("2026-03-12", "infraestrutura", "O asfalto todo esburacado da avenida principal está causando acidentes. Precisa de recapeamento urgente, os carros desviam dos buracos e invadem a outra pista."),
    ("2026-03-12", "educação", "O ônibus escolar da zona rural quebrou e as crianças estão faltando às aulas porque não têm como chegar à escola."),
    ("2026-03-13", "segurança", "Assaltos frequentes no ponto de ônibus da Avenida Norte, principalmente de manhã cedo quando ainda está escuro e os trabalhadores esperam a condução."),
    ("2026-03-13", "meio ambiente", "O córrego que passa atrás do Conjunto Habitacional Primavera está recebendo esgoto clandestino. A água ficou preta e tem cheiro forte."),
    ("2026-03-16", "educação", "O telhado da quadra da escola estadual está caindo e os alunos estão proibidos de fazer educação física há três meses."),
    ("2026-03-16", "saúde", "Falta atendimento no PSF do meu bairro. Não tem doutor para consultar e mandam a gente voltar outro dia, mas no outro dia também não tem."),
    ("2026-03-17", "meio ambiente", "Um bar na Rua Alagoas coloca som muito alto até as três da manhã, de quinta a domingo. Ninguém na rua consegue dormir."),
    ("2026-03-17", "segurança", "Escrevo em nome da associação de moradores do Parque das Árvores. Há cerca de seis meses se instalou um ponto de venda de drogas na praça ao lado da escola municipal, e desde então aumentaram os furtos de celulares, as brigas e as pichações. A praça está sem iluminação, o que facilita a ação dos criminosos, e a viatura da Guarda Municipal só aparece quando alguém liga, o que demora mais de uma hora. Os pais estão com medo de deixar os filhos irem sozinhos para a aula. Solicitamos rondas regulares no horário de entrada e saída dos alunos, a instalação de câmeras e a iluminação da praça."),
    ("2026-03-18", "saúde", "A ambulância do SAMU demorou mais de uma hora para chegar quando meu pai passou mal em casa com dor no peito."),
    ("2026-03-18", "infraestrutura", "Poste sem luz na Rua Pernambuco. À noite a rua vira um breu e ninguém consegue ver nada, já pedimos a troca das lâmpadas várias vezes."),
    ("2026-03-19", "educação", "Faltam vagas na creche municipal. Estou na fila de espera há oito meses e preciso voltar a trabalhar."),
    ("2026-03-19", "meio ambiente", "Queimadas frequentes no terreno ao lado da escola estão deixando as crianças com problemas respiratórios e os olhos ardendo."),
    ("2026-03-20", "educação", "As crianças da Escola Paulo Freire não estão recebendo lanche há vários dias porque a merenda acabou e não chegou reposição."),
    ("2026-03-20", "saúde", "Não consigo marcar exame de ultrassom pelo SUS. A fila está com mais de seis meses de espera e meu médico pediu com urgência."),
    ("2026-03-23", "infraestrutura", "Lâmpada queimada na praça do bairro Boa Vista. A praça fica escura e perigosa à noite e os moradores deixaram de usar o espaço."),
    ("2026-03-23", "segurança", "Motos fazendo racha na Avenida Beira Rio nos fins de semana. Já houve um atropelamento e os moradores têm medo de atravessar."),
    ("2026-03-24", "meio ambiente", "Todas as semanas caminhões descarregam entulho de construção e restos de poda às margens do Rio Jaguaribe, no trecho próximo à ponte da BR. O material já forma uma barreira que desvia a água e, na última cheia, o rio transbordou para dentro das casas da Rua da Beira. Além do entulho há sacos de lixo doméstico, pneus velhos e até carcaças de geladeira. Os peixes estão morrendo e os pescadores da colônia não conseguem mais tirar o sustento. Já denunciamos à Secretaria de Meio Ambiente e nada foi feito. Pedimos fiscalização com multa para quem descarta irregularmente e a limpeza do leito do rio."),
    ("2026-03-24", "saúde", "Faltam remédios de pressão na farmácia do posto. Tenho que comprar do meu bolso e sou aposentado com um salário mínimo."),
    ("2026-03-25", "meio ambiente", "Terreno abandonado na Rua Goiás cheio de lixo e entulho. Aparecem ratos e baratas nas casas vizinhas e ninguém limpa."),
    ("2026-03-25", "educação", "Os computadores do laboratório de informática da escola estão quebrados há mais de um ano e os alunos não têm aula de informática."),
    ("2026-03-26", "infraestrutura", "O semáforo do cruzamento da Av. Brasil com a Rua Sergipe está desligado há dias e já houve três batidas."),
    ("2026-03-26", "segurança", "Todo dia alguém é roubado na parada de ônibus da Av. Norte logo cedo. Precisamos de policiamento nesse horário."),
    ("2026-03-27", "saúde", "Minha mãe tem 78 anos e precisa de fisioterapia depois de uma cirurgia, mas o posto diz que não tem profissional disponível."),
    ("2026-03-27", "educação", "Meu filho tem 9 anos, é cadeirante e estuda na Escola Municipal Santa Luzia. A escola não tem rampa na entrada, o único banheiro adaptado está trancado servindo de depósito e a sala dele fica no primeiro andar, sem elevador. Todos os dias um funcionário precisa carregá-lo no colo pela escada, o que é humilhante e perigoso. Além disso, desde fevereiro ele está sem a professora auxiliar que a lei garante, e a professora da turma não consegue dar a atenção de que ele precisa. Já protocolei pedidos na Secretaria de Educação duas vezes. Peço obras de acessibilidade e a contratação imediata da auxiliar de inclusão."),
]

# Pares de duplicatas semânticas (mesmo problema, palavras diferentes)
DUPLICATAS = [("M003", "M017"), ("M005", "M026"), ("M008", "M022"),
              ("M011", "M029"), ("M014", "M035"), ("M019", "M038")]

if __name__ == "__main__":
    base = Path(__file__).parent
    registros = [{"id": f"M{i + 1:03d}", "data": d, "categoria_oficial": c, "texto": t}
                 for i, (d, c, t) in enumerate(M)]
    assert len(registros) == 40
    assert all(50 <= len(r["texto"]) <= 800 for r in registros), [len(r["texto"]) for r in registros]
    longos = [r["id"] for r in registros if len(r["texto"]) > 500]
    assert len(longos) == 5, longos
    (base / "manifestacoes.json").write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "duplicatas_gabarito.json").write_text(
        json.dumps([list(p) for p in DUPLICATAS], indent=2), encoding="utf-8")
    print("OK: 40 manifestações | longas:", longos, "| duplicatas:", len(DUPLICATAS))
