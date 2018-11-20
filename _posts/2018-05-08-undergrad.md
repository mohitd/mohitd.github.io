---
title: "Undergraduate Tips and Tricks"
excerpt: "After finishing finish my undergrad, I'll share some pointers (pun absolutely intended) that I've learned over the past few years."
comments: true
---

I recently finished my last year as an undergrad, and I thought I'd share some neat tips and tricks that I've accumulated over the years in the hopes that someone will find some of these points useful.

_DISCLAIMER: I was a computer science and engineering major at a fairly large university (Ohio State University), but I'll try my best to generalize these points._

# General

---

**Always get a full night's sleep**. There have been [studies](http://www.sciencedirect.com/science/article/pii/0301051178900315) and [articles](https://pdfs.semanticscholar.org/8501/fa541d13634b021a7cab8d1d84ba2c5b9a7c.pdf) that show REM sleep, i.e., the deepest kind of sleep, helps with memory consolidation and committing to long-term memory. For example, being well-rested for several days before an exam helps the brain commit the exam content to long-term memory.

Sleep also reduces fatigue from staring at the same block of code trying to find a bug. There were countless times where I spent hours trying to fix a bug, gave up for the night, and fixed the bug within 5 minutes the next morning.

**Make use of every free resource you can get**. Money can be scarce for undergrads so make full use of whatever free things you can get! This might include free prints at computer labs, free software, and especially free food! I used to keep a printer in my dorm/apartment for my first two years, but, with the number of free prints in the computer labs, I never had to use it. I know that you're technically paying thousands of dollars for those "free" prints, but you might as well make full use of all of the facilities and programs that your tuition gives you.

Having an edu email address also opens the door to a ton of free or discounted stuff. When I was really into mobile UI/UX design some years ago, I ran into an app called Sketch that I really liked. As it turns out, they offer educational discounts so I managed to get it at 50% off, and I wasn't even using it for a class! Always poke around for a student discount before buying anything.

**Don't overcommit**. There are a myriad of new and exciting things going on during your college years: hackathons, side-projects, research, and clubs. Trying to do them all will detrimentally affect your sleep or classes. When you take on too many responsibilities, you end up doing a poor job at all of them. Something like missing a deadline should be a clear, blinking signal that you need to cut back. Many enthusiastic people will approach you with opportunities, and the real challenge is knowing when to politely decline.

I've been guilty of this in the past. I've completely forgotten to submit assignments or attend lectures because I was busy doing something else. Every so often students, professors, or businesspeople ask me to help with a task, join a group, or build a product, and I have to take a mental inventory of what I'm currently involved to come to an answer.

> Never half-ass two things. Whole-ass one thing. - Ron Swanson

**Read**. If you want to become a better tennis player, watch and play tennis. If you want to become a better writer, read and write more. Regardless of your major, you'll have to do some kind of writing. It might not be flowing, elegant prose, but it has to be coherent and grammatically correct. In STEM fields, you'll have to do more technical writing, but even technical writing needs to be cohesive, i.e., you can't just string facts together into paragraphs and weave a report like that. (Although, your first few attempts will certainly look like that!) I've found that reading fiction can really help any writing style, especially for technical writing, since it helps you think in terms of constructing a narrative story. All good technical works give motivation and elaborate on related work (Exposition), explain their approach (Rising Action), and present results in that narrative (Climax and Resolution). By the way, for superior scientific writing, check out [Writing Science](https://www.amazon.com/Writing-Science-Papers-Proposals-Funded/dp/0199760241).

Of course, you don't have to read just for the purpose of becoming a better writer. I don't believe that anyone really _hates_ to read; I think they just haven't found a genre or topic they find interesting enough. I didn't used to like it or make time for it because most students were forced to read prose from the early 19th and 20th centuries. But then I explored a bit more and found authors and genres that I genuinely enjoyed. (And many I didn't!) If you're not a fan of books, try reading shorter articles. I usually read 4-5 articles on [Medium](http://www.medium.com) per day.

Reading is just a simple, general-purpose activity that you can do every day that will transform your way of thinking and make you a smarter person overall.

**Take a day off every so often**. You need to take some time to recharge. As a student, you probably have a ton of stuff to do all the time, but taking a day off is equally important: it prevents burnout. Going at a high-output pace all the time causes the quality of that output to suffer, or it takes much longer to produce the same quality output. Taking a day off helps mentally recharge, and you'll notice your work quality will improve after some rest.

**If you're thinking about grad school, do research**. You should certainly get involved in research if you're going for a PhD since prior research experience is the [most important thing](http://www.pgbovine.net/grad-school-app-tips.htm). Even if you want a Masters, having prior research experience is a strong point on your application. Take a look at my other [post on undergrad research to help you get started](/undergrad-research)!

**Maintain a website**. You're probably reading this post on my website. I highly encourage making a site that highlights personal projects (maybe have a link to a demo or a video of a demo), open-source contributions, work experience, and involvement in organizations or clubs. Your department probably has some means to host a site on their servers; if not, check out Github Pages. Your website serves as a portfolio of your undergrad. Just remember to keep it updated!

Also on my site, you'll notice I have other blog posts that are more explanatory in nature, e.g., backpropagation and Restricted Boltzmann Machines (RBMs). I also highly encourage writing about topics you're familiar with or, even better, topics you _aren't_ familiar with. When I was writing the post on RBMs, I was actually trying to learn about them. So I started from the formulation and worked my way through, writing down the material in a way that made sense to me. If I had a question about anything, I looked it up and added it to the post. In the end, I wrote a complete post that I can refer to if needed, and that post may very well be the first that another student learning the material reads.

Also, don't be discouraged by experts. I'm positive there are many people much smarter than I when it comes to backpropagation or RBMs and have already written about them many times over. But not everyone learns in the same way, so expecting everyone to learn from the same slide deck or article is ludicrous! Having a variety of explanations at different levels about a topic is bound to help someone learn.

# Studying

---

**Re-derive what you're learning from scratch**.
> What I cannot create, I do not understand. - Richard Feynman  
> Know how to solve every problem that has been solved. - Richard Feynman

While it's a bit of a hyperbole, re-deriving an equation or process from scratch gives you a _much_ better understand of the topic. If you're in computer science, you'll probably take some theory classes (theory of computation/automata, neural networks, algorithms, etc.). When introduced to a new topic in these classes, try to go through the derivation yourself. Reading through a derivation or proof is _completely different_ than doing it a blank piece of paper by yourself.

This doesn't necessarily have to be for a class. For example, when I was learning about neural networks, I always thought of backpropagation as some mystic black-box that magically computes all of the gradients to train the network. But I kept running into issues when I started to train my own neural networks so I delved deeper and re-derived the backpropagation equations from scratch. It took quite a bit of time, and I made many errors along the way, but the insight I gained after successfully completing the derivation was invaluable.

(Side note: if you're interested in AI at all, I highly recommend building your own neural network library from scratch since it gives you deeper understanding than using someone else's library where the details are abstracted away from you.)

**Teach others what you know**. Be willing to share your knowledge. If you're doing well in class, try hosting group study sessions for midterms and finals. The stronger students in the study group will help provide more insight, and the weaker students will ask more fundamental questions ("why do we use X for Y?"). I've found that teaching others really helps you understand the material as well as the students you are teaching: it helps paint a cohesive picture about the topic you're teaching.

Always ask "why" and "how" questions, e.g., "why do we need backpropagation for training a deep network?" The "what" and "when" questions are simply definitions that you can look up on your own time. But the "why" and "how" questions are the most important because they demonstrate true understanding. In other words, when you ask "why" and "how" questions, you're not just learning the mechanics of a concept but rather you understand why the mechanics are the way that they are and not something else.

**Your study environment matters**. If you prefer to study at home, you have to deal with far more potential avenues of distraction than if you were studying at the university library or your department's lab/lounge. A more subtle reason for considering your study environment is that your brain make weak associations with your environment and memory. This phenomenon is called state-dependent memory: recall is a little bit easier if you are in the same state of conscientiousness as when you studied. (Note that I _did not_ say that state-dependent memory is guaranteed or perfect!) With this in mind (pun absolutely intended), studying in a few different places helps create redundant weak associations. (For those familiar with AI, this is kind of like dropout used in neural networks, except for your brain!) Obviously, you shouldn't depend on this psychology solely and actually study 🙂

**Attend review sessions**. In my experience, review sessions tend to be quite polarized: they're either really useful or quite useless. I've sat through review sessions where the professor reads through the text on each and every lecture slide; these were probably less productive sessions. There were also question-based review sessions where the professor just stood at the front of the classroom and did nothing until someone asked a question; these were somehow worse since much of the time was spent in absolute silence. On the other hand, I've also attended review sessions that were mostly about solving example problems; these were the most helpful, but not every class is problem-based like mathematics or theory.

In all cases, go to the review session anyways, even if your professor reads off of his slides. It'll help you remember the material covered in the first few weeks. There's also a good chance that other students will ask questions that you may not have thought of at the time. Also, you'll be exposed to the material once again, and that will help reinforce it.

**Make a review sheet**. Even if you're not allowed to bring a cheat sheet, make one anyways. Writing down the core concepts of the course helps reinforce the topics in your brain. It also doubles as a one/two/three/n-page sheet that sums up the course's content that you can refer to at any point instead of having to look through pages and pages of your notes or the textbook. This same sheet is something you can use to help guide study sessions.

In the event you _are_ allowed to bring a cheat sheet, then add worked-through example problems as well as formulas. The exam is probably going to have actual problems on it, not just regurgitating formulas, so including some example problems will help jog your memory on _how_ to solve the problems.

**Start with example midterms/review sheets before studying**. This has two purposes: it gets you in the mindset of the class, and, most importantly, it helps assess what topics you should focus your studying efforts. People have the tendency to solve practice problems that they know how to do as some assurance of their knowledge retention. They shy away from the difficult problems simply because they're difficult, but working through the difficult problems provides the most insight. If you can solve the most difficult problem in the homework or on the practice midterm, then you'll most likely do well on the actual exam. That being said, just solving difficult problems leads the risk of forgetting how to do the simple stuff. So, like many things, it's all about balance!

**Prefer concreteness to abstractness**. Many professors, TAs, and students will try to teach topics starting from the most general definitions since they tend to be the most complete and comprehensive. However, I find it easier to do a concrete example. For instance, backpropagation is a particular topic in neural networks that many students struggle to understand. It's either presented with pages of equations or as a magic black-box. So if you're having a hard time understanding how it works, do a small example by hand, and it will help your understanding. You'll know how all of the theory combines together to get to your final concrete answer. Abstractions are difficult to understand at first since we live in a concrete world; do an example or problem that adds concrete numbers to the abstract concept.

# Exams

---

**Eat something before the exam**. This piece of advice is a bit peculiar but important nonetheless. Never take an exam on an empty stomach but also don't take an exam in a food coma. Eat before the exam so your food has time to settle. Hunger is not something you want to be focusing on during the exam!

**Get some sleep the night before**. Staying up the night before cramming is probably not going to do you any good. Remember that during REM sleep is when your brain consolidates and processes memories. Without it, your brain can't consolidate the studying you've done over the past day. Certainly study the day before the exam but don't stay up all night!

**You don't have to take the test linearly**. Nothing says you have to take a test sequentially (for most exams, anyways). I usually take the first few minutes to go through each page and solve the problems that I can do or know quickly. By doing this first pass, you knock out all of the quick and easy questions and get a feel for the more difficult ones since you've seen them, so now you can focus your time on the harder questions. The second pass is for the more difficult questions you may need to think about more.

**Always double-check your work**. Double-check the entire exam, even the problems that you thought were really easy. Especially be sure to double-check the more difficult and multi-step problems as well. I used to be notorious for missing a few points here or there for little math mistakes, and it adds up!

**Never leave the test early**. Take your time! Similar to the advice about double-checking your work, triple-check and quadruple-check your work. There's probably something you missed or forgot. Even if you're absolutely 100% sure of everything! Leaving early is just leaving points on the table.

**Communicate to your grader**. The point of the exam is to show your professor or grader that you know the material. Convey that message. Even if you don't know exactly how to solve a problem, at least write down what you would do or how you would proceed. But don't write an essay on a problem. That leaves more chance for the grader to find a mistake and take off points. This also makes the grader unhappy, and an unhappy grader is not good. Make your grader happy by doing clean work and communicating.

# Attending Class

---

**Attend class. (unless you think your time is better spent elsewhere)**. Avoid skipping class in general, but spend your time better when you do. In other words, don't skip class just to play video games. There will be times where you'll be very comfortable with a topic, and the professor is lecturing on it for a week. If you're slammed with work from other classes or a paper deadline, those should probably be prioritized, but still take a look at the slide deck or grab the notes just in case.

At one point, I was well-versed in C; I was working on several C projects, school and personal. I was also really getting into the weeds with my research and had a few big undergrad research events where I had to prepare posters, abstracts, and presentations. I was taking a required class and saw on the syllabus that the first month was being spent on introductory C. Instead of going to class and learning about C all over again, I spent my time working on research. The professor read off of the slides anyways, and they were posted online. That being said, I still spoke with classmates about lectures in case the professor highlighted some key points that would be on the exam. And I still did go to the review session to recap and learn what was going to be on the midterm specifically. But in general, please try to attend class 🙂

**Prefer handwritten notes to typed notes**. This is a bit of a stylistic point, but there's [evidence](http://drawingchildrenintoreading.com/assets/the_pen_is_mightier_than_the_keyboard-libre.pdf) that suggests handwritten notes promote memory retention more than typed notes. I take all of my notes using just paper and pencil. While this can be a bit cumbersome at times, I certainly remember more of what I write down than what I type. If you're not a note-taking Luddite like me, I've seen other students use handwrite notes on tablets or computers with touchscreens.

**Write it down even if you think it's obvious**. One big mistake students make is "I don't need to write down that obvious fact. I'll just remember it." No, you won't. You will not remember it. Even if it's obvious write it down anyways. It might make perfect sense while you're in the middle of lecture, but the point is that it should make sense _after_ lecture. There were several times I had that same erroneous thought process, and, when I went back to review my notes, I had no idea what I wrote down.

**Be wary of whom you work with on group projects**. Group projects are the bane of every student, from the highly motivated student that is genuinely interested in learning the material to the student that just wants to get the class over with. I've been both, and I suspect many of you have as well. Assuming you have the choice, select your group members wisely! If you don't have a choice, then you just have to work with what you have!

---

**Time is a non-renewable resource**. Grades do not define you. Find a balance between "never coming to class" and "spending every waking minute studying". Neither of these extremes will be beneficial to you.

Carve out time to study and for side projects. Create something! It's your creations that leave a mark on the world! It's much easier at a interview (job, grad school, etc.) to discuss your robotics side project or Android app or machine learning model than how easy it was to get an "A" in your architecture class. Your interviewers are probably far more interested in the former than the latter.

Good luck!
