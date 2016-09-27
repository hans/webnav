This is the first instantiation of a paradigm in **simulated language acquisition**
that I have been developing during my time at OpenAI.

### The paradigm

Here's the gist, for those interested:

- A child exists in some physical world. The child has limited observations and
  a limited action space. (i.e., there are observations or actions necessary to
  achieve the goal which the child cannot perform).
- There is also a **parent** in the same environment which has full observation and
  a full action space over the environment.
- The parent speaks some fixed language, and takes actions or provides
  information only when requested by the child.
- The child can send messages to the parent and observe the parent's responses.

This paradigm is designed such that the child *must learn the language of the
parent in order to accomplish the goal.* Crucially, language here is a
**side-effect** of the quest to achieve some other non-linguistic goal.

### This application

This repository contains a concrete application of the idea above. Here our
"child" is spawned on a random page in the Wikipedia graph, and must navigate
to some target page only by clicking on page links. Of course, our goal is to
make the traversal in the minimum number of hops. This is a pretty difficult
task requiring a substantial amount of world knowledge. It becomes easier,
though, when you have a "parent" around to help!

There are a lot of possible ways to define a "parent" in this context. At the
simplest level, a parent might point out the best link to follow (according to
e.g. an A\* search or a heuristic). At the most complex level, a parent might
simply Google the child's questions or send them to a knowledge base, and then
forward the responses on to the child.

#### Example trajectory

Here's a trajectory from a Q-learning model in this environment. The child
always has the choice to 1) take actions in its environment or 2) communicate
with the parent. Here we can see the learned policy alternates between the two.

Notes:

- The child can utter a single token at each timestep (shown as `"string"`
  actions below), visit a new page, or `SEND` the tokens uttered so far to
  the mother. It only receives a response after executing `SEND`.
- There is a fixed rollout length; after reaching the target page, this child
  learns to make short "cycles" around the page.
- The numbers in parentheses indicate the index of the action in the ordered
  list of possible actions. The mother responds with a number string, which
  the child learns to correctly map onto its own action space (as demonstrated
  by these numbers in parentheses).

```
Trajectory: (target Frédéric_Chopin)

        Action                                          Reward
        -------------------------------------------------------
        Boeing_747 (0)                                  0.00000
        "which"                                        -0.25000
        SEND                                            0.72850
                --> Response: "6"
        Piano (6)                                       0.77280
        Frédéric_Chopin (16)                            8.28645
        Warsaw (6)                                      0.17489
        "which"                                        -0.25000
        SEND                                            0.72850
                --> Response: "15"
        Jazz (15)                                       0.07623
        "which"                                        -0.25000
        SEND                                            0.72850
                --> Response: "12"
        Piano (12)                                      0.77280
        Frédéric_Chopin (6)                             8.28645
        Niccolò_Paganini (25)                           0.84642
        Frédéric_Chopin (3)                             8.28645
```
