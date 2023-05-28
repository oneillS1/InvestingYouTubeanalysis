

"""
This is a text file outlining the criteria and logic used for various parts of the project

Part 3: Finding relevant videos

Criteria for inclusion:
    Relevance:
        Channel found via the search method(s)
            1. Search by keywords on YouTube - using #notfinancialadvice, crypto to buy, stocks to buy, what to invest in
            2. Snowball method from Google search of most important investing influencers on YouTube
        Video itself covers investing (any product)
        Video or channel over X views (X TO BE DECIDED ON VIEWING THE LIST OF VIDEOS)

    Pragmatics - must meet the following criteria:
        Transcript available on YouTube
        Between 2 - 10-minute videos (may also do shorts if possible)

Relevant channels from Google search:

Stocks:
    Ricky Gutierrez
    The Trading Channel: Offers technical analysis, trading strategies, and stock picks.
    Warrior Trading: Focuses on day trading strategies, stock market education, and occasional stock picks.
    Stock Moe: Provides stock market analysis, investing tips, and stock recommendations.
    Charlie Chang: Offers insights on long-term investing, stock analysis, and investment strategies.
    Alex Becker's Channel: Covers stock market trends, investing strategies, and occasional stock recommendations.
    ZipTraderU: Provides stock market education, trading strategies, and stock picks.
    Informed Trades: Offers educational content on stock trading, technical analysis, and investing tips.
    ClayTrader: Focuses on stock trading education, technical analysis, and occasional stock picks.
    Trade Ideas: Provides stock scanning tools, trading strategies, and market analysis.
    Financial Education 3: Offers stock market insights, investing tips, and occasional stock picks.
    Financial Education: Offers stock market analysis, investing strategies, and recommendations.
    Meb Faber Research: Provides insights on global investing, asset allocation, and stock market trends.
    The Motley Fool: Covers stock recommendations, investing strategies, and market insights.
    Ryan Scribner: Offers stock market analysis, investment tips, and occasional stock recommendations.
    Jeremy Lefebvre - ClayTrader: Focuses on stock trading, technical analysis, and stock picks.
    Ricky Gutierrez: Provides day trading strategies, stock analysis, and occasional stock picks.
    ZipTrader: Offers technical analysis, stock picks, and market insights.
    Investing with Tom: Covers value investing, stock analysis, and occasional stock recommendations.
    Financial Education 2: Provides stock market news, analysis, and stock picks.
    Learn to Invest: Offers investment analysis, stock recommendations, and investing strategies.
    Meet Kevin:
    Graham Stephen:
    Andrei Jikh:
    Nate O'Brien:
    Wealth Hacker:
    Wolf of Dubai Stocks Investing Channel:
    Financial Education: Offers stock market analysis, investing strategies, and stock recommendations.
    Meb Faber Research: Provides insights on global investing, asset allocation, and stock market trends.
    The Motley Fool: Covers stock recommendations, investing strategies, and market insights.
    Ryan Scribner: Offers stock market analysis, investment tips, and occasional stock recommendations.
    Jeremy Lefebvre - ClayTrader: Focuses on stock trading, technical analysis, and stock picks.
    Ricky Gutierrez: Provides day trading strategies, stock analysis, and occasional stock picks.
    ZipTrader: Offers technical analysis, stock picks, and market insights.
    Investing with Tom: Covers value investing, stock analysis, and occasional stock recommendations.
    Financial Education 2: Provides stock market news, analysis, and stock picks.
    Learn to Invest: Offers investment analysis, stock recommendations, and investing strategies.


Crypto:
        @BitBoyCryptoChannel; @SheldonEvansx; @CryptoBanterGroup; @TheCryptoLark;
        Coin Bureau: Offers in-depth analysis, reviews, and recommendations on various cryptocurrencies.
        DataDash: Covers cryptocurrency market analysis, trading strategies, and coin reviews.
        Ivan on Tech: Provides educational content on blockchain technology, cryptocurrencies, and market insights.
        Crypto Jebb: Offers technical analysis, market updates, and cryptocurrency investment strategies.
        Altcoin Daily: Focuses on daily cryptocurrency news, market updates, and long-term investment opportunities.
        Crypto Zombie: Covers cryptocurrency news, project reviews, and market analysis.
        Chico Crypto: Provides in-depth analysis, project reviews, and investment tips on cryptocurrencies.
        BitBoy Crypto: Offers cryptocurrency market analysis, project reviews, and investment strategies.
        Crypto Love: Covers cryptocurrency news, project updates, and investment advice.
        The Modern Investor: Provides insights on cryptocurrencies, market analysis, and investment strategies.
        Crypto Capital Venture: Offers technical analysis, project reviews, and cryptocurrency market insights.
        Crypto Daily: Focuses on cryptocurrency news, market updates, and investment opportunities.
        CryptosRUs: Provides cryptocurrency market analysis, project reviews, and investment strategies.
        Nugget's News: Covers cryptocurrency news, market updates, and educational content.
        Crypto Crow: Offers insights on cryptocurrencies, project reviews, and investment tips.
        Crypto Beadles: Provides interviews, project reviews, and investment advice on cryptocurrencies.
        Ready Set Crypto: Focuses on technical analysis, market insights, and cryptocurrency investment strategies.
        The Moon: Offers technical analysis, market updates, and investment opportunities in cryptocurrencies.
        Crypto Gurus: Provides cryptocurrency news, project reviews, and investment analysis.
        Cryptocurrency Market: Covers market updates, project reviews, and investment strategies in cryptocurrencies.
        Brian Jung:

Usernames of channels above
    Crypto (27):
            "@SheldonEvansx", "@CryptoBanterGroup", "@TheCryptoLark", @BitBoyCryptoChannel, @CoinBureau
            @DataDash, @IvanOnTech, @CryptoJebb, @AltcoinDaily, @CryptoZombie, @CryptoLove,
            @TheModernInvestor, @CryptoCapitalVenture, @CryptoDaily, @CryptosRUs, @NuggetsNews, @CryptoCrowOfficial,
            @cryptobeadles3949, @ReadySet, @TheMoon, @JamesCryptoGuru, @Jungernaut,

    Stocks (24):
            @RickyGutierrezz, @thetradingchannel, @DaytradeWarrior, @StockMoe, @CharlieChang, @AlexBeckersChannel,
            @ZipTrader, @InformedTrades, @claytrader, @TradeIdeas, @FinancialEducation, @MotleyFool, @RyanScribner
            @InvestingwithTom, @LearntoInvest, @themebfabershow1017, @MeetKevin, @GrahamStephan,
            @AndreiJikh, @NateOBrien, @wealthhacker, @wolfofdubai, @StockswithJosh, @jeremylefebvremakesmoney7934

Part 4: Webscraping

What data to include:
    Channel
    Date
    Title
    Description
    Video length
    Number of likes & dislikes
    Number of comments

    Transcript text
        Autogenerated / Creator written

"""