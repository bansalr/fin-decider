# Labelling guide (ontology 1.1.1)

Label each article from its text alone. Model outputs are deliberately hidden.

## Columns

- `gsib_activity`, `market_context_only`, `neither`: `y` or `n` each. More than one can be `y` (an article can describe bank activity *and* market context).
- `markets`, `lending`, `treasury_alm`: `y`/`n`, only when `gsib_activity` is `y`; otherwise leave blank.
- `leaves`: the codes of every activity category below that the article materially describes, separated by `;`. Leave blank if none.
- `confidence`: 1 (unsure) to 3 (certain). `notes`: anything ambiguous. `labeller`: your initials.

Rule of thumb: the activity must actually happen in the article (a deal, trade, loan, facility, or a bank business line's results), not just be mentioned. Central banks, bank analysts' forecasts, and price moves alone are market context, not G-SIB activity.

## Top level

**gsib_activity**: Does the article describe a commercial or investment bank, dealer, or lender doing business, such as lending, underwriting, trading, market making, financing clients, or managing its own funding? Central-bank actions, bank analysts' forecasts or commentary, and price moves of bonds, currencies, commodities, or stocks do not count.
- yes: A bank or dealer's lending, deal, trade, client financing, or funding activity is reported, or a bank business line's results or strategy are discussed.
- no: Only prices, economic data, central-bank or government policy, bank analysts' forecasts or commentary, or non-bank company news, with no bank or dealer business activity described.

**market_context_only**: Is the article about financial markets, the economy, policy, or companies, including central-bank actions and bank analysts' forecasts or commentary, without describing a commercial or investment bank or dealer lending, underwriting, trading for clients, or financing?
- yes: Market moves, prices, economic data, central-bank or government policy, bank analysts' forecasts or commentary, or company news, with no bank or dealer business activity described.
- no: A bank or dealer activity is described, or the article is not about markets or the economy at all.

**neither**: Is the article unrelated to financial markets, the economy, and banking?
- yes: General news such as sport, culture, crime, weather, or non-financial human-interest stories.
- no: Any financial-market, economic, policy, company, or banking content is discussed.

## Activity categories (codes for `leaves`)

- `treasury.funding_liquidity`: **Funding & Liquidity**. Does the article describe a bank raising funding for its own balance sheet or managing its own liquidity, for example issuing its own bonds or building its liquidity buffer?
- `treasury.structural_risk`: **Structural Rate & FX Risk**. Does the article describe a bank managing or hedging the interest-rate or currency risk on its own balance sheet?
- `markets.macro.rates`: **Rates**. Does the article describe a bank or dealer trading or making markets in government bonds, interest-rate swaps, inflation-linked bonds, or other rates products, including in emerging markets?
- `markets.macro.fx`: **Foreign Exchange**. Does the article describe a bank or dealer trading or making markets in currencies for clients, including emerging-market currencies, FX forwards, swaps, or options?
- `markets.macro.commodities`: **Commodities**. Does the article describe a bank or dealer trading commodities or providing commodity derivatives such as futures or swaps?
- `markets.credit.cash`: **Corporate Bond Trading**. Does the article describe a bank or dealer trading or making markets in corporate bonds, including investment-grade, high-yield (junk), or distressed debt?
- `markets.credit.credit_derivatives`: **Credit Derivatives & Structured Credit**. Does the article describe a bank or dealer trading or structuring credit default swaps, credit indices, or tranched credit products such as synthetic CDOs?
- `markets.equities.cash`: **Cash Equities**. Does the article describe a bank or broker executing or trading stocks for clients or making markets in shares, such as block trades?
- `markets.equities.derivatives`: **Equity Derivatives & Structured Equity**. Does the article describe a bank or dealer trading or structuring equity options, equity swaps, index derivatives, or equity-linked structured notes?
- `markets.securitised.mbs`: **Mortgage-Backed Securities**. Does the article describe a bank or dealer creating, trading, or selling residential or commercial mortgage-backed securities?
- `markets.securitised.abs`: **ABS**. Does the article describe a bank or dealer creating, trading, or selling asset-backed securities such as auto-loan or credit-card ABS?
- `markets.securitised.clo_cdo`: **CLO / CDO**. Does the article describe a bank or dealer creating, warehousing, trading, or selling collateralised loan obligations or CDOs?
- `markets.financing.repo`: **Repo / Reverse Repo**. Does the article describe a bank or dealer providing repo or reverse-repo financing to clients against securities?
- `markets.financing.prime_services`: **Prime Services & Securities Finance**. Does the article describe a bank providing prime brokerage, margin loans, securities lending, or total-return-swap financing to hedge funds or other investors?
- `markets.derivatives_xva`: **Derivatives / XVA**. Does the article describe a bank or dealer providing OTC or cleared derivatives, clearing services for clients, or managing derivative valuation adjustments such as CVA?
- `markets.underwriting.dcm`: **Debt Capital Markets Underwriting**. Does the article describe banks underwriting, arranging, or running the books on a bond sale?
- `markets.underwriting.ecm`: **Equity Capital Markets Underwriting**. Does the article describe banks underwriting or running the books on an IPO, share sale, or convertible bond offering?
- `markets.underwriting.loan_syndication`: **Loan Syndication**. Does the article describe banks arranging or syndicating a loan across multiple lenders?
- `lending.corporate.large_mid`: **Large & Mid-Sized Corporate Lending**. Does the article describe a bank lending to a large or mid-sized company through a general-purpose loan, term loan, or revolving credit facility?
- `lending.corporate.sme`: **SME Corporate Lending**. Does the article describe a bank lending to small or medium-sized businesses?
- `lending.leveraged_sponsor.leveraged_loan`: **Leveraged Loan**. Does the article describe banks providing leveraged loans to highly indebted or below-investment-grade companies, such as term loan B?
- `lending.leveraged_sponsor.acquisition`: **Acquisition & Bridge Finance**. Does the article describe banks providing loans, bridge loans, or committed financing to fund an acquisition, merger, or buyout?
- `lending.specialised.project_finance`: **Project Finance**. Does the article describe lenders financing a specific project, such as a power plant, pipeline, or infrastructure, repaid from the project's own revenues?
- `lending.specialised.object_finance`: **Object Finance**. Does the article describe a bank financing a specific aircraft, ship, railcar, or other physical asset, with repayment tied to that asset?
- `lending.cre.general`: **Commercial Property Lending**. Does the article describe a bank making or holding loans secured on commercial property such as offices, shopping centres, hotels, or rental buildings?
- `lending.cre.hvcre`: **Construction & Development Lending**. Does the article describe a bank financing the acquisition, development, or construction of commercial real estate?
- `lending.financial_institutions`: **Financial Institutions Lending**. Does the article describe a bank lending to other financial firms, such as banks, broker-dealers, insurers, finance companies, or investment funds?
- `lending.sovereign_public`: **Sovereign / Public Sector Lending**. Does the article describe a bank lending to a government, sovereign, or public-sector body?
- `lending.trade_working_capital`: **Trade, Working-Capital & Commodity Finance**. Does the article describe a bank providing trade finance, letters of credit, guarantees, supply-chain or receivables financing, or financing of commodity inventories, reserves, or trade flows?
- `lending.consumer`: **Consumer Lending**. Does the article describe a bank or lender providing credit to individuals through credit cards, car loans, or personal loans?
- `lending.residential_mortgage`: **Residential Mortgage Lending**. Does the article describe a bank or lender providing home loans secured on residential property?
