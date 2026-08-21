import uuid

from agent.agent_state import AgentState


class SessionManager:
    """
    Manage Swiftrail customer conversations.
    """


    def __init__(self):

        # Store active sessions
        self.sessions = {}



    def create_session(self, customer_id=None, customer_name=""):
        """
        Create a new customer session.
        """

        session_id = str(uuid.uuid4())


        state = AgentState(
            session_id=session_id
        )


        # Attach customer information
        if customer_id:

            state.set_customer(
                customer_id,
                customer_name
            )


        self.sessions[session_id] = state


        return state



    def get_session(self, session_id):
        """
        Retrieve existing session.
        """

        return self.sessions.get(session_id)



    def update_session(self, session_id, note):
        """
        Add information during conversation.
        """

        session = self.get_session(session_id)


        if session:

            session.add_note(note)



    def end_session(self, session_id):
        """
        Close customer session.
        """

        if session_id in self.sessions:

            del self.sessions[session_id]



    def list_sessions(self):
        """
        Return active sessions.
        """

        return list(self.sessions.keys())
