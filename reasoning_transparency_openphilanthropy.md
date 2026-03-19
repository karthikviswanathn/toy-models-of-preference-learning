# Reasoning Transparency (Open Philanthropy excerpt)

> Editor’s note: This article was published under our former name, Open Philanthropy. Some content may be outdated. You can see our latest writing here.

We at the Open Philanthropy Project value analyses which exhibit strong “reasoning transparency.” This document explains what we mean by “reasoning transparency,” and provides some tips for how to efficiently write documents that exhibit greater reasoning transparency than is standard in many domains.

In short, our top recommendations are to:

- Open with a linked summary of key takeaways.
- Throughout a document, indicate which considerations are most important to your key takeaways.
- Throughout a document, indicate how confident you are in major claims, and what support you have for them.

## Motivation

When reading an analysis — e.g. a scientific paper, or some other collection of arguments and evidence for some conclusions — we want to know: “How should I update my view in response to this?” In particular, we want to know things like:

- Has the author presented a fair or biased presentation of evidence and arguments on this topic?
- How much expertise does the author have in this area?
- How trustworthy is the author in general? What are their biases and conflicts of interest?
- What was the research process that led to this analysis? What shortcuts were taken?
- What rough level of confidence does the author have in each of their substantive claims?
- What support does the author think they have for each of their substantive claims?
- What does the author think are the most important takeaways, and what could change the author’s mind about those takeaways?
- If the analysis includes some data analysis, how were the data collected, which analyses were done, and can I access the data myself?

Many scientific communication norms are aimed at making it easier for a reader to answer questions like these, e.g. norms for ‘related literature’ sections and ‘methods’ sections, open data and code, reporting standards, pre-registration, conflict of interest statements, and so on.

In other ways, typical scientific communication norms lack some aspects of reasoning transparency that we value. For example, many scientific papers say little about roughly how confident the authors are in different claims throughout the paper, or they might cite a series of papers (or even entire books!) in support of specific claims without giving any page numbers.

Below, I (Luke Muehlhauser) offer some tips for how to write analyses that (I suspect, and in my experience) make it easier for the reader to answer the question, “How should I update my views in response to this?”

## Example of GiveWell charity reviews

I’ll use a GiveWell charity review to illustrate a relatively “extreme” model of reasoning transparency, one that is probably more costly than it’s worth for most analysts. Later, I’ll give some tips for how to improve an analysis’ reasoning transparency without paying as high a cost for it as GiveWell does.

Consider GiveWell’s review of Against Malaria Foundation (AMF). This review…

- includes a summary of the most important points of the review, each linked to a longer section that elaborates those points and the evidence for them in some detail.
- provides detailed responses to major questions that bear on the likely cost-effectiveness of marginal donations to AMF, e.g. “Are LLINs targeted at people who do not already have them?”, “Do LLINs reach intended destinations?”, “Is there room for more funding?”, and “How generally effective is AMF as an organization?”
- provides a summary of the research process GiveWell used to evaluate AMF’s cost-effectiveness.
- provides an endnote, link to another section or page, or other indication of reasoning/sources for nearly every substantive claim. There are 125 endnotes, and in general, the endnote provides the support for the corresponding claim, e.g. a quote from a scientific paper, or a link to a series of calculations in a spreadsheet, or a quote from a written summary of an interview with an expert. (There are some claims that do not have such support, but these still tend to clearly signal what the basis for the claim is; e.g. “Given that countries and other funders have some discretion over how funds will be used, it is likely that some portion of AMF’s funding has displaced other funding into other malaria interventions and into other uses.”)
- provides a comprehensive table of sources, including archived copies of most sources in case some of the original links break at some point.
- includes a list of remaining open questions about AMF’s likely cost-effectiveness, plus comments throughout the report on which claims about AMF GiveWell is more or less confident in, and why.
- links to a separate summary of the scientific evidence for the effectiveness of the intervention AMF performs, namely the mass distribution of long-lasting insecticide-treated nets (LLINs), which itself exhibits all the features listed above.
- plus much more

## Most important recommendations

Most analysts and publishers, including the Open Philanthropy Project,[1] don’t (and shouldn’t) invest as much effort as GiveWell does to achieve improved reasoning transparency. How can you improve an analysis’ reasoning transparency cheaply and efficiently? Below are some tips, with examples in the text and in footnotes.

### Open with a linked summary of key takeaways

Many GiveWell and Open Philanthropy Project analyses open with a summary of key takeaways, with links to later sections that elaborate and argue for each of those key takeaways at greater length (examples in footnote[2]). This makes it easy for the reader to understand the key takeaways and examine particular takeaways in more depth.

### Indicate which considerations are most important

Which arguments or pieces of evidence are most critical for your key takeaways? Ideally, this should be made clear early in the document, or at least early in the section discussing each key takeaway.

Some of my earlier Open Philanthropy Project reports don’t do this well. E.g. my carbs-obesity report doesn’t make it clear that the evidence from randomized controlled trials (RCTs) played the largest role in my overall conclusions.

Some examples that do a better job of this include:

- My report on behavioral treatments for insomnia makes clear that my conclusions are based almost entirely on RCT evidence, and in particular on (1) the inconclusive or small results from RCTs that “tested [behavioral treatments] against a neutral control at ≥1mo follow-up using objective measures of [total sleep time or sleep efficiency],” (2) the apparent lack of any “high-quality, highly pragmatic” RCTs on the question, and (3) my general reasons for distrusting self-report measures of sleep quality.
- After surveying a huge variety of evidence, my report on consciousness and moral patienthood provides an 8-point high-level summary that makes it (somewhat) clear how I’ve integrated those diverse types of evidence into an overall conclusion, and which kinds of evidence play which roles in that conclusion.
- The introduction of Holden’s blog post on worldview diversification makes it clear which factors seem to be most important in the case one could make for or against using a worldview diversification strategy, and the rest of the post elaborates each of those points.
- Holden’s Potential Risks from Advanced Artificial Intelligence: The Philanthropic Opportunity does the same.

### Indicate how confident you are in major claims, and what support you have for them

For most substantive claims, or at least for “major” claims that are especially critical for your conclusions, try to give some indication of how confident you are in each claim, and what support you think you have for it.

#### Expressing degrees of confidence

Confidence in a claim can be expressed roughly using words such as “plausible,” “likely,” “unlikely,” “very likely,” and so on. When it’s worth the effort, in some cases you might want to express your confidence as a probability or a confidence interval, in part because terms like “plausible” can be interpreted differently by different readers.

Below are examples that illustrate the diversity of options available for expressing degrees of confidence with varying precision and varying amounts of effort:[3]

- “I think there is a nontrivial likelihood (at least 10% with moderate robustness, and at least 1% with high robustness) of transformative AI within the next 20 years” (source). This is a key premise in the case for making potential risks from advanced AI an Open Philanthropy Project priority, so we thought hard about how confident we should be that transformative AI would be created in the next couple decades, and decided to state our confidence using probabilities.
- In a table, I reported specific probabilities that various species (e.g. cows, chickens) are phenomenally conscious, because such estimates were a major goal of the investigation. However, I also made clear that my probabilities are hard to justify and may be unstable, using nearby statements such as “I don’t have much reason to believe my judgments about consciousness are well-calibrated” and “I have limited introspective access to the reasons why my brain has produced these probabilities rather than others” and “There are many different kinds of uncertainty, and I’m not sure how to act given uncertainty of this kind.” (The last of these is followed by a footnote with links to sources explaining what I mean by “many kinds of uncertainty.”)
- “…my own 70% confidence interval for ‘years to [high-level machine intelligence]’ is something like 10–120 years, though that estimate is unstable and uncertain” (source). This is the conclusion to a report about AI timelines, so it seemed worthwhile to conclude the report with a probabilistic statement about my forecast — in this case, in terms of a 70% confidence interval — but I also indicate that this estimate is “unstable and uncertain” (with a footnote explaining what I mean by that).
- “It is widely believed, and seems likely, that regular, high-quality sleep is important for personal performance and well-being, as well as for public safety and other important outcomes” (source). This claim was not a major focus of the report, so I simply said “seems likely,” to indicate that I think the probability of my statement being true is >50%, while also indicating that I haven’t investigated the evidence in detail and haven’t tried to acquire a more precise probabilistic estimate.
- “CBT-I appears to be the most commonly-discussed [behavioral treatment for insomnia] in the research literature, and is plausibly the most common [behavioral treatment for insomnia] in clinical practice” (source). I probably could have done 1-3 hours of research and become more confident about whether CBT-I is the most common behavioral treatment for insomnia in clinical practice, but this claim wasn’t central to my report, so instead I just reported my rough impression after reading some literature, and thus reported my level of confidence as “plausible.”
- “The rise of bioethics seems to be a case study in the transfer of authority over a domain (medical ethics) from one group (doctors) to another (bioethicists), in large part due to the first group’s relative neglect of that domain” (source). Here, I use the phrase “seems to be” to indicate that I’m fairly uncertain even about this major takeaway, and the context makes clear that this uncertainty is (at least in part) due to the fact that my study of the history of bioethics was fairly quick and shallow.
- In my report on behavioral treatments for insomnia, I expressed some key claims in colloquial terms in the main text, but in a footnote provided a precise, probabilistic statement of my claim. For example, I claimed in the main text that “I found ~70 such [systematic reviews], and I think this is a fairly complete list,” and my footnote stated: “To be more precise: I’m 70% confident there are fewer than 5 [systematic reviews] on this topic, that I did not find, published online before October 2015, which include at least 5 [randomized controlled trials] testing the effectiveness of one or more treatments for insomnia.”
- “I have raised my best estimate of the chance of a really big storm, like the storied one of 1859, from 0.33% to 0.70% per decade. And I have expanded my 95% confidence interval for this estimate from 0.0–4.0% to 0.0–11.6% per decade” (source). In this case, the author (David Roodman) expressed his confidences precisely, since his estimates are the output of a statistical model.

#### Indicating kinds of support

Given limited resources, you cannot systematically and carefully examine every argument and piece of evidence relevant to every claim in your analysis. Nor can you explain in detail what kind(s) of support you think you have for every claim you make. Nevertheless, you can quickly give the reader some indication of what kind(s) of support you have for different claims, and you can explain in relatively more detail the kind(s) of support you think you have for some key claims.

Here are some different kinds of support you might have for a claim:

- another detailed analysis you wrote
- careful examination of one or more studies you feel qualified to assess
- careful examination of one or more studies you feel only weakly able to assess
- shallow skimming of one or more studies you feel qualified to assess
- shallow skimming of one or more studies you feel only weakly able to assess
- verifiable facts you can easily provide sources for
- verifiable facts you can’t easily provide sources for
- expert opinion you feel comfortable assessing
- expert opinion you can’t easily assess
- a vague impression you have based on reading various sources, or talking to various experts, or something else
- a general intuition you have about how the world works
- a simple argument that seems robust to you
- a simple argument that seems questionable to you
- a complex argument that nevertheless seems strong to you
- a complex argument that seems questionable to you
- the claim seems to follow logically from other supported claims plus general background knowledge
- a source you can’t remember, except that you remember thinking at the time it was a trustworthy source, and you think it would be easy to verify the claim if one tried[4]
- a combination of any of the above

Below, I give a series of examples for how to indicate different kinds of support for different claims, and I comment briefly on each of them.

Here’s the first example:

> “As stated above, my view is based on a large number of undocumented conversations, such that I don’t think it is realistic to aim for being highly convincing in this post. Instead, I have attempted to lay out the general structure of the inputs into my thinking. For further clarification, I will now briefly go over which parts of my argument I believe are well-supported and/or should be uncontroversial, vs. which parts rely crucially on information I haven’t been able to fully share…” [source]

In some cases, you can’t provide much of the reasoning for your view, and it’s most transparent to simply say so.

Next example:

> “It is widely believed, and seems likely, that regular, high-quality sleep is important for personal performance and well-being, as well as for public safety and other important outcomes.” [source]

This claim isn’t the focus of the report, so I didn’t review the literature on the topic, and my phrasing makes it clear that I believe this claim merely because it is “widely believed” and “seems likely,” not because I have carefully reviewed the relevant literature.

> “[CBT-I] can be delivered on an individual basis or in a group setting, via self-help (with or without phone support), and via computerized delivery (with or without phone support).” [source]

This claim is easily verifiable, e.g. by Googling “computerized CBT-I” or “self-help CBT-I,” so I didn’t bother to explain what kind of support I have for it — other than to say, in a footnote, “I found many or several RCTs testing each of these types of CBT-I.”

> “PSG is widely considered the ‘gold standard’ measure of sleep, but it has several disadvantages. It is expensive, complicated to interpret, requires some adaptation by the patient (people aren’t used to sleeping with wires attached to them), and is usually (but not always) administered at a sleep lab rather than at home.” [source]

These claims are substantive but non-controversial, so as support I merely quote (in a footnote) some example narrative reviews which back up my claims.

> “Supposedly (I haven’t checked), [actigraphy] correlates well with PSG on at least two key variables…” [source]

Here, I quote from narrative reviews in a footnote, but because the “correlates well with” claim is more substantive and potentially questionable/controversial than the earlier claims about PSG, I flag both my uncertainty, and the fact that I haven’t checked the primary studies, by saying “Supposedly (I haven’t checked)…” This is a good example of a phrasing that helps improve a document’s reasoning transparency, but would rarely be found in e.g. a scientific paper.

> “these seven trials were only moderately pragmatic in design.” [source]

To elaborate on this claim, in a footnote I make a prediction about the results of a very specific test for “pragmaticness.” The footnote both makes it clear what I mean by “moderately pragmatic in design,” and provides a way for someone to check whether my statement about the studies is accurate. Again, the idea isn’t that the reader can assume my predictions are well-calibrated, but rather that I’m being clear about what I’m claiming and what kind of support I think I have for it. (In this case, my support is that I skimmed the papers and I came away with an intuition that they wouldn’t score very highly on a certain test of study pragmaticness.)

Also, I couldn’t find anything succinct that clearly explained what I meant by “pragmatic,” and the concept of pragmaticness was fairly important to my overall conclusions in the report, so I took the time to write a separate page on that concept, and linked to that page from this report.

> “I have very little sense of how much these things would cost. My guess is that if a better measure of sleep (of the sort I described) can be developed, it could be developed for $2M-$20M. I would guess that the “relatively small” RCTs I suggested might cost $1M-$5M each, whereas I would guess that a large, pragmatic RCT of the sort I described could cost $20M-$50M. But these numbers are just pulled from vague memories of conversations I’ve had with people about how much certain kinds of product development and RCT implementation cost, and my estimates could easily be off by a large factor, and maybe even an order of magnitude.” [source]

Here, I’m very clear that I have no basis whatsoever for the cost estimates I provide.

> “both a priori reasoning about self-report measures and empirical reviews of the accuracy of self-report measures (across multiple domains) lead me to be suspicious of self-reported measures of sleep.” [source]

In this case, I hadn’t finished my review of the literature on self-report measures, and I didn’t have time to be highly transparent about the reasoning behind my skepticism of the accuracy of self-report measures, so in a footnote I simply said: “I am still researching the accuracy of self-report measures across multiple domains, and might or might not produce a separate report on the topic. In the meantime, I only have time to point to some of the sources that have informed my preliminary judgments on this question, without further comment or argument at this time. [Long list of sources.] Please keep in mind that this is only a preliminary list of sources: I have not evaluated any of them closely, they may be unrepresentative of the literature on self-report as a whole, and I can imagine having a different impression of the typical accuracy of self-report measures if and when I complete my report on the accuracy of self-report measures. My uncertainty about the eventual outcome that investigation is accounted for in the predictions I have made in other footnotes in this report…” This is another example of a footnote that improves the reasoning transparency of the report, but is a paragraph you’d be unlikely to read in a journal article.

> “I will, for this report, make four key assumptions about the nature of consciousness. It is beyond the scope of this report to survey and engage with the arguments for or against these assumptions; instead, I merely report what my assumptions are, and provide links to the relevant scholarly debates. My purpose here isn’t to contribute to these debates, but merely to explain ‘where I’m coming from.’” [source]

Here, I’m admitting that I didn’t take the time to explain the support I think I have for some key assumptions of the report.

> “As far as we know, the vast majority of human cognitive processing is unconscious, including a large amount of fairly complex, ‘sophisticated’ processing. This suggests that consciousness is the result of some particular kinds of information processing, not just any information processing. Assuming a relatively complex account of consciousness, I find it intuitively hard to imagine how (e.g.) the 302 neurons of C. elegans could support cognitive algorithms which instantiate consciousness. However, it is more intuitive to me that the ~100,000 neurons of the Gazami crab might support cognitive algorithms which instantiate consciousness. But I can also imagine it being the case that not even a chimpanzee happens to have the right organization of cognitive processing to have conscious experiences.” [source]

Here, I make it clear that my claims are merely intuitions.

> “By the time I began this investigation, I had already found persuasive my four key assumptions about the nature of consciousness: physicalism, functionalism, illusionism, and fuzziness. During this investigation I studied the arguments for and against these views more deeply than I had in the past, and came away more convinced of them than I was before. Perhaps that is because the arguments for these views are stronger than the arguments against them, or perhaps it is because I am roughly just as subject to confirmation bias as nearly all people seem to be (including those who, like me, know about confirmation bias and actively try to mitigate it). In any case: as you consider how to update your own views based on this report, keep in mind that I began this investigation as a physicalist functionalist illusionist who thought consciousness was likely a very fuzzy concept.” [source]

This example makes it clear that the reader shouldn’t conclude that my investigation led me to these four assumptions, but instead that I already had those assumptions before I began.

> “Our understanding is that it is not clearly possible to create an advanced artificial intelligence agent that avoids all challenges of this sort. [footnote:] Our reasoning behind this judgment cannot be easily summarized, and is based on reading about the problem and having many informal conversations. Bostrom’s Superintelligence discusses many possible strategies for solving this problem, but identifies substantial potential challenges for essentially all of them, and the interested reader could read the book for more evidence on this point.” [source]

This is another example of simply saying “it would be too costly to summarize our reasoning behind this judgment, which is based on many hours of reading about the topic and discussing it with others.”

Often, a good way to be transparent about the kind of support you think you have for a claim is to summarize the research process that led to the conclusion. Examples:

- “Here is a table showing how the animals I ranked compare on these factors (according to my own quick, non-expert judgments)… But let me be clear about my process: I did not decide on some particular combination rule for these four factors, assign values to each factor for each species, and then compute a resulting probability of consciousness for each taxon. Instead, I used my intuitions to generate my probabilities, then reflected on what factors seemed to be affecting my intuitive probabilities, and then filled out this table” (source).
- “I spent less than one hour on this rapid review. Given this limitation, I looked only for systematic reviews released by the Cochrane Collaboration…, a good source of reliably high-quality systematic reviews of intervention effectiveness evidence. I also conducted a few Google Scholar keyword searches to see whether I could find compelling articles challenging the Cochrane reviews’ conclusions, but I did not quickly find any such articles” (source).
- “I did not conduct any literature searches to produce this report. I have been following the small field of HLMI forecasting closely since 2011, and I felt comfortable that I already knew where to find most of the best recent HLMI forecasting work” (source).
- “To investigate the nature of past AI predictions and cycles of optimism and pessimism in the history of the field, I read or skim-read several histories of AI and tracked down the original sources for many published AI predictions so I could read them in context. I also considered how I might have responded to hype or pessimism/criticism about AI at various times in its history, if I had been around at the time and had been trying to make my own predictions about the future of AI… I can’t easily summarize all the evidence I encountered that left me with these impressions, but I have tried to collect many of the important quotes and other data below” (source).
- “To find potential case studies on philanthropic field-building, I surveyed our earlier work on the history of philanthropy, skimmed through the many additional case studies collected in The Almanac of American Philanthropy, asked staff for additional suggestions, and drew upon my own knowledge of the history of some fields. My choices about which case studies to look at more closely were based mostly on some combination of (1) the apparent similarity of the case study to our mid-2016 perception of the state of the nascent field of research addressing potential risks from advanced AI (the current focus area of ours where the relevant fields seem most nascent, and where we’re most likely to apply lessons from this investigation in the short term), and (2) the apparent availability and helpfulness of sources covering the history of the case study. I read and/or skimmed the sources listed in the annotated bibliography below, taking notes as I went. I then wrote up my impressions (based on these notes) of how the relevant fields developed, what role (if any) philanthropy seemed to play, and anything else I found interesting. After a fairly thorough look at bioethics, I did quicker and more impressionistic investigations and write-ups on a number of other fields” (source).

For further examples, see the longer research process explanations in David Roodman’s series on the effects of incarceration on crime; my reports on the carbs-obesity hypothesis and behavioral treatments for insomnia; the “our process” sections of most Open Philanthropy Project cause reports and GiveWell top charity reviews and intervention reports; and a variety of other reports.[5]

## Secondary recommendations

### Provide quotes and page numbers when possible

When citing some support for a claim, provide a page number if possible. Even better if you can directly quote the most relevant passage, so the reader doesn’t need to track down the source to get a sense of what kind of support for the claim the source provides.

Especially if your report is published online, there are essentially no space constraints barring you from including dozens or hundreds of potentially lengthy quotes from primary sources in footnotes and elsewhere. E.g. see the many long quotes in the footnotes of my report on consciousness and moral patienthood, or the dozens of quotes in the footnotes of GiveWell intervention reports.

### Provide data and code when possible

Both GiveWell and Open Philanthropy Project provide underlying data, calculations, and code when possible.

In some cases these supplementary materials are fairly large, as with the 800mb of data and code for David Roodman’s investigation into the impact of incarceration on crime, or the 11-sheet cost-effectiveness models (v4) for GiveWell’s top charities.

In other cases they can be quite small, for example a 27-item spreadsheet of case studies I considered examining for Some Case Studies in Early Field Growth, or a 14-item spreadsheet of global catastrophic risks.

### Provide archived copies of sources when possible

Both GiveWell and Open Philanthropy Project provide archived copies of sources when possible,[6] since web links can break over time.

### Provide transcripts or summaries of conversations when possible

For many investigations, interviews with domain experts will be a key source of information alongside written materials. Hence when possible, it will help improve the reasoning transparency of a report if those conversations (or at least the most important ones) can be made available to the reader, either as a transcript or a summary.

But in many cases this is too time-costly to be worth doing, and in many cases a domain expert will only be willing to speak frankly with you anonymously, or will only be willing to be quoted/cited on particular points.
